import numpy as np
import os
from lspi.sample import Sample

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from plot_traj_sagittal import plot_leg_sagittal_plane
# import pykeyboard # Ensure this library is installed and compatible with your environment
from pynput import keyboard


def visualize_gait_cycle(cfg, file, data_list):
    plot_data(file, cfg, data_list,["LOADCELL"], ylim=[-30, 140])
    plot_data(file, cfg, data_list,["ACTUATOR_POSITION"], ylim=[-60,130])
    plot_data(file, cfg, data_list,["SHANK_ANGLE"], ylim=[-60,110])
    plot_data(file, cfg, data_list,["GAIT_PHASE"], ylim=[-0.1,1.1])
    plot_data(file, cfg, data_list,["GAIT_SUBPHASE"], ylim=[-0.1,4.1])
    plot_data(file, cfg, data_list,[f"LINEAR_ACC_{i}_LOCAL" for i in ["X","Y","Z"]], ylim=[-30, 30], mark_max_min=False)
    plot_data(file, cfg, data_list,[f"GRAVITY_VECTOR_{i}" for i in ["X","Y","Z"]], ylim=[-10, 20], mark_max_min=False)
    plot_data(file, cfg, data_list, ["ACTUATOR_SETPOINT","TORQUE_ESTIMATE"], ylim=[-110, 110] )

def plot_data(file, cfg, data_list, names, ylim, 
              plot_mean_std = True, 
              mark_max_min = True,
              show = False):
    fig = plt.figure(figsize=(10, 5))
    
    for name in names:
        for i in range(len(data_list)):    
            plt.scatter(np.arange(0,1,0.01), data_list[i][name], s=1.5, c="grey")#, label=f"gait cycle {i+1}", alpha=0.2)
        
        # if mark_max_min_individual:
        #     for i in range(len(data_list)):
        #         max_idx = np.argmax(data_list[i][name])
        #         min_idx = np.argmin(data_list[i][name])
        #         plt.scatter(max_idx/100, data_list[i][name].max(), s=10, c="red", marker="o")
        #         plt.scatter(min_idx/100, data_list[i][name].min(), s=10, c="blue", marker="x")
        
        if plot_mean_std:
            mean = np.mean(np.array([data[name] for data in data_list]), axis=0)
            std = np.std(np.array([data[name] for data in data_list]), axis=0)
            plt.plot(np.arange(0,1,0.01), mean, label=f"mean {name.lower()}")
            plt.fill_between(np.arange(0,1,0.01), mean-3*std, mean+3*std, alpha=0.2)
            # mean_phase_change =  mean_st_sw_phase(data_list, loadcell_threshold = cfg["loadcell_threshold"])
            mean_phase_change = np.mean([np.where(data["GAIT_PHASE"])[0][0] for data in data_list])
            plt.axvline(mean_phase_change / 100, color='green', linestyle='--')
            plt.text(mean_phase_change / 100, ylim[1] - 0.05 * (ylim[1] - ylim[0]), 
                     f"{mean_phase_change/100:.2f}", fontsize=10, color="green")
            if not mark_max_min:
                
                min_idx_before = np.argmin(mean[:int(mean_phase_change)])
                max_idx_before = np.argmax(mean[:int(mean_phase_change)]) # restrict max to happened before min
                max_idx_after = np.argmax(mean[int(mean_phase_change):]) + int(mean_phase_change)
                min_idx_after = np.argmin(mean[int(mean_phase_change):]) + int(mean_phase_change)

                plt.scatter(max_idx_before/100, mean[max_idx_before], s=15, c="black", marker="o")
                plt.scatter(min_idx_before/100, mean[min_idx_before], s=15, c="black", marker="x")
                plt.text(max_idx_before/100, mean[max_idx_before] + 0.05 * (ylim[1] - ylim[0]), 
                         f"max: {mean[max_idx_before]:.2f} at {max_idx_before/100:.2f}", fontsize=10, color="black")
                plt.text(min_idx_before/100, mean[min_idx_before] - 0.05 * (ylim[1] - ylim[0]), 
                         f"min: {mean[min_idx_before]:.2f} at {min_idx_before/100:.2f}", fontsize=10, color="black")

                plt.scatter(max_idx_after/100, mean[max_idx_after], s=15, c="black", marker="o")
                plt.scatter(min_idx_after/100, mean[min_idx_after], s=15, c="black", marker="x")
                plt.text(max_idx_after/100, mean[max_idx_after] + 0.05 * (ylim[1] - ylim[0]), 
                         f"max: {mean[max_idx_after]:.2f} at {max_idx_after/100:.2f}", fontsize=10, color="black")
                plt.text(min_idx_after/100, mean[min_idx_after] - 0.05 * (ylim[1] - ylim[0]), 
                         f"min: {mean[min_idx_after]:.2f} at {min_idx_after/100:.2f}", fontsize=10, color="black")
        
    plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.grid()
    plt.xlabel("phase")
    plt.ylabel("/".join(names).lower())
    os.makedirs("plot", exist_ok=True)
    plt.title(f"{file[:-4]}")
    plt.savefig(f"plot/{'_'.join(names)}_{file[:-4]}.png".lower())
    if show:
        plt.show()
    plt.close()

# def mean_st_sw_phase(data_list, loadcell_threshold = 0.5):
#     return np.mean([np.where(np.where(data_list[i]["LOADCELL"] < loadcell_threshold, 1, 0))[0][0] for i in range(len(data_list))])

def skip_outliers(data, sw_st_idx, cfg, variable_name, test = False):
    """
    Skip the outliers in the data
    by using the mean and std of the selected variable
    return the list of data for each gait cycle, and the mean and std of the selected variable
    """
    data_list = []
    
    for i in range(len(sw_st_idx)-1):
        # skip the first few gait cycles
        if i < cfg["num_skip_gait_cycles"]:
            continue
        # skip the gait cycles that shorter than 0.5s
        if (sw_st_idx[i+1] - sw_st_idx[i]) < cfg[ "min_timesteps_per_gait_cycle"]:
            print(f"skip gait cycle {i} because of too short")
            continue
        
        interpolated_data = data.iloc[sw_st_idx[i]:sw_st_idx[i+1]].copy()
        interpolated_data = interpolated_data.interpolate(method='linear', limit_direction='both')
        # interpolate the data to 100 points
        interpolated_data = interpolated_data.iloc[np.linspace(0, len(interpolated_data)-1, 100).astype(int)]
        
        # # skip the gait cycles that stance and swing are inproportional
        if np.sum(interpolated_data["GAIT_PHASE"]) >  70 or np.sum(interpolated_data["GAIT_PHASE"]) < 30:
            print(f"skip gait cycle {i} because of inproportional phase")
            continue
        
        if test:
            print("Entering test mode")
            # plot the data by frame in sagittal plane
            # st_ratio = np.sum(interpolated_data["LOADCELL"] > cfg["loadcell_threshold"]) / len(interpolated_data)
            st_ratio = np.sum(1 - interpolated_data["GAIT_PHASE"]) / len(interpolated_data)
            plot_leg_sagittal_plane(interpolated_data["ACTUATOR_POSITION"] -20, interpolated_data["SHANK_ANGLE"], st_ratio=st_ratio, skip_rate=10)
            
        # np.save( "knee_angle.npy", interpolated_data["ACTUATOR_POSITION"])
        # np.save( "shank_angle.npy", interpolated_data["SHANK_ANGLE"])
        data_list.append(interpolated_data)
    
    if len(data_list) > 0: 
        data_KA = np.array(pd.concat(data_list, axis=0)[variable_name]).reshape(-1, 100)
        
        mean, std = np.mean(data_KA, axis=0), np.std(data_KA, axis=0)
        # skip the knee angle outside mean +- 3* std
        mask = np.all(np.abs(data_KA - mean) <= 3 * std, axis=1)
        
        data_list_skip = [data_list[i] for i in range(len(data_list)) if mask[i]]

        return data_list_skip
    else:
        print(f"no data")
        return []



def extract_gait_data(f,cfg, filter_kernel_size = 5):
    """
    Extract gait data from the given file.
    Filter the LOADCELL data using a moving average filter.
    Infer the stance or swing phase based on the filtered LOADCELL data.
    return the data with filtered LOADCELL data and the SW to ST index of the gait cycle (start of each gait cycle)
    """
    # print(f)
    raw_data = pd.read_csv(f, header=0, skiprows=0)
                # Convert numeric columns to float, ignoring non-numeric data
    raw_data = raw_data.apply(pd.to_numeric, errors='coerce')
                # Drop rows with any NaN values resulting from non-numeric data
    raw_data = raw_data.dropna()    
    raw_data = raw_data[(raw_data > -1e6).all(axis=1) & (raw_data < 1e6).all(axis=1)]
    
    filtered_loadcell = raw_data["LOADCELL"] #np.convolve(raw_data["LOADCELL"].values, np.ones(filter_kernel_size)/filter_kernel_size, mode='same')
    raw_data["LOADCELL"] = filtered_loadcell
    # infer the stance or swing phase by loadcell data:
    # phase = np.where(filtered_loadcell >= cfg["loadcell_threshold"], 1, 0)
    phase = 1 - raw_data["GAIT_PHASE"].values
    
    # phase = np.where(raw_data["GAIT_SUBPHASE"].values >= 1, 0, 1) 
    # phase = np.where(raw_data["GAIT_SUBPHASE"].values <= 2, 1, 0) 
    sw_st_idx = np.where(phase[:-1] - phase[1:] == -1)[0] + 1
    st_sw_idx = np.where(phase[:-1] - phase[1:] == 1)[0] + 1

    phase_var = np.zeros(len(phase))
    for i in range(len(sw_st_idx)-1):
        phase_var[sw_st_idx[i]:sw_st_idx[i+1]] = np.linspace(0, 1, sw_st_idx[i+1] - sw_st_idx[i])
                
    # clip to keep only complete gait cycles
    data, phase_var, phase = raw_data.iloc[sw_st_idx[0]:sw_st_idx[-1]], phase_var[sw_st_idx[0]:sw_st_idx[-1]], phase[sw_st_idx[0]:sw_st_idx[-1]]
    sw_st_idx = sw_st_idx[1:] - sw_st_idx[0]
    st_sw_idx = st_sw_idx[st_sw_idx > sw_st_idx[0]] - sw_st_idx[0]
    return data,sw_st_idx, st_sw_idx

        
def add_quadratic_reward_stack(ssa_samples, cfg, w_s = 0.8):
    """
    Add a quadratic reward stack to the samples.
    The reward is calculated as:
    reward = -1/2 * ||s - s_target||^2 * w_s - 1/2 * ||a||^2 * w_a
    where w_s and w_a are the weights for the state and action respectively.
    """
    n_total = ssa_samples.shape[1]
    n_action = len(cfg["control_variables"])
    n_state = (n_total - n_action)//2
    
    # init reward stack 
    s_err = ssa_samples[:,:n_state] 
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