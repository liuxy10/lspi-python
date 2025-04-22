import numpy as np
import os
from lspi.sample import Sample

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


def visualize_gait_cycle(cfg, file, data_list, mean, std):
    fig = plt.figure(figsize=(10, 5))
    for i in range(len(data_list)):
        # plt.scatter(np.linspace(0,1,100), data_list[i]["KNEE_ANGLE"], label=f"gait cycle {i+1}", s=1)
        # plt.plot(np.linspace(0,1,100), data_list[i]["GAIT_SUBPHASE"]*15)
        plt.scatter(np.linspace(0,1,100), data_list[i]["LOADCELL"])#, label=f"gait cycle {i+1}", alpha=0.2)
    plt.ylim(-30, 140)
    plt.xlabel("phase")
    plt.ylabel("LOADCELL (N)")
    os.makedirs("plot", exist_ok=True)
    plt.savefig(f"plot/loadcell_{file[:-4]}.png")

    fig = plt.figure(figsize=(10, 5))
    for i in range(len(data_list)):
        plt.scatter(np.linspace(0,1,100), data_list[i]["KNEE_ANGLE"], label=f"gait cycle {i+1}", s=1)
        # plt.plot(np.linspace(0,1,100), data_list[i]["GAIT_SUBPHASE"]*15)
    plt.plot(np.linspace(0,1,100), mean, label="mean", color="black")
    plt.fill_between(np.linspace(0,1,100), mean-3*std, mean+3*std, alpha=0.2, label="mean +- 3* std")
    plt.xlabel("time")
    plt.ylabel("KNEE_ANGLE (degree)")
    plt.title(f"parameters: {[v['name'] for v in cfg['control_variables']]}={file}")
    plt.ylim(-60,130)
    plt.grid()
    plt.savefig(f"plot/knee_angle_{file[:-4]}.png")

    plt.close()

def skip_outliers(data, sw_st_idx, cfg):
 
    data_list = []
    
    for i in range(len(sw_st_idx)-1):
        # skip the first few gait cycles
        if i < cfg["num_skip_gait_cycles"]:
            continue
        # skip the gait cycles that shorter than 0.5s
        if (sw_st_idx[i+1] - sw_st_idx[i]) < cfg[ "min_timesteps_per_gait_cycle"]:
            continue
        
        interpolated_data = data.iloc[sw_st_idx[i]:sw_st_idx[i+1]].copy()
        interpolated_data = interpolated_data.interpolate(method='linear', limit_direction='both')
        # interpolate the data to 100 points
        interpolated_data = interpolated_data.iloc[np.linspace(0, len(interpolated_data)-1, 100).astype(int)]
        data_list.append(interpolated_data)
    
    data_KA = np.array(pd.concat(data_list, axis=0)["KNEE_ANGLE"]).reshape(-1, 100)
    mean, std = np.mean(data_KA, axis=0), np.std(data_KA, axis=0)
    # skip the knee angle outside mean +- 3* std
    mask = np.all(np.abs(data_KA - mean) <= 3 * std, axis=1)
    
    data_list_skip = [data_list[i] for i in range(len(data_list)) if mask[i]]

    return data_list_skip, mean, std



def extract_gait_data(f,cfg, filter_kernel_size = 5):
    raw_data = pd.read_csv(f, header=0, skiprows=0)
                # Convert numeric columns to float, ignoring non-numeric data
    raw_data = raw_data.apply(pd.to_numeric, errors='coerce')
                # Drop rows with any NaN values resulting from non-numeric data
    raw_data = raw_data.dropna()    
    raw_data = raw_data[(raw_data > -1e6).all(axis=1) & (raw_data < 1e6).all(axis=1)]
    
    filtered_loadcell = np.convolve(raw_data["LOADCELL"].values, np.ones(filter_kernel_size)/filter_kernel_size, mode='same')
    raw_data["LOADCELL"] = filtered_loadcell
    # infer the stance or swing phase by loadcell data:
    phase = np.where(filtered_loadcell >= cfg["loadcell_threshold"], 1, 0)
    
    # phase = np.where(raw_data["GAIT_SUBPHASE"].values >= 1, 0, 1)
    # phase = np.where(raw_data["GAIT_SUBPHASE"].values <= 2, 1, 0)
    sw_st_idx = np.where(phase[:-1] - phase[1:] == -1)[0] + 1


    phase_var = np.zeros(len(phase))
    for i in range(len(sw_st_idx)-1):
        phase_var[sw_st_idx[i]:sw_st_idx[i+1]] = np.linspace(0, 1, sw_st_idx[i+1] - sw_st_idx[i])
                
    # clip to keep only complete gait cycles
    data, phase_var, phase = raw_data.iloc[sw_st_idx[0]:sw_st_idx[-1]], phase_var[sw_st_idx[0]:sw_st_idx[-1]], phase[sw_st_idx[0]:sw_st_idx[-1]]
    sw_st_idx = sw_st_idx[1:] - sw_st_idx[0]
    return data,sw_st_idx

        
def add_quadratic_reward_stack(ssa_samples, cfg, s_target = np.zeros(4), w_s = 0.8):
    n_total = ssa_samples.shape[1]
    n_action = len(cfg["control_variables"])
    n_state = (n_total - n_action)//2
    
    # init reward stack 
    s_err = ssa_samples[:,:n_state] - s_target
    assert w_s >= 0 and w_s <= 1, "w_s should be between 0 and 1"
    w_a = np.sqrt(1 - w_s**2)
    rew = - 1/2 * np.sum(s_err**2, axis=1) * w_s - 1/2 * np.sum(ssa_samples[:,-n_action:]**2, axis=1) * w_a
    rew = rew.reshape(-1, 1)
    # print(f"rew.shape", rew.shape)
    
    return np.concatenate([ssa_samples, rew], axis=1)

def load_from_data(ssar, n_state = 4, n_action = 3):
    samples = []
    assert n_state *2 + n_action + 1 == ssar.shape[1], "ssar should have shape (n_samples, n_state*2 + n_action + 1)"
    for i in range(len(ssar)):
        state = ssar[i][:n_state]
        next_state = ssar[i][n_state:n_state*2]
        action = ssar[i][n_state*2:-1]
        reward = ssar[i][-1]
        done = False
        samples.append(Sample(state, action, reward, next_state, done))
    return samples

def load_samples_from_file(filename):
    samples = []
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            state = np.array([float(x) for x in data[0:4]])
            action = int(data[4])
            reward = float(data[5])
            next_state = np.array([float(x) for x in data[6:10]])
            done = bool(int(data[10]))
            samples.append(Sample(state, action, reward, next_state, done))
    return samples

def generate_file(file_name, num_samples=1000):
    with open(file_name, 'w') as file:
        for _ in range(num_samples):
            state = np.random.rand(4)
            action = np.random.randint(0, 2)
            reward = np.random.rand()
            next_state = np.random.rand(4)
            done = np.random.choice([0, 1])
            file.write(f"{','.join(map(str, state))},{action},{reward},{','.join(map(str, next_state))},{done}\n")

def generate_samples(states, params, num_samples=1000):
    pairs_idx = np.random.choice(np.arange(len(params)), size=(3000, 2), replace=True)
    
    print(f"params: {params}, pairs_idx: {pairs_idx}")
    samples = []
    for _ in range(num_samples):
        state = np.random.rand(4)
        action = np.random.randint(0, 2)
        reward = np.random.rand()
        next_state = np.random.rand(4)
        done = np.random.choice([0, 1])
        samples.append(Sample(state, action, reward, next_state, done))
    return samples