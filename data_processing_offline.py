import numpy as np
import os

from lspi.lspi_train_offline import lspi_loop_offline
from lspi.solvers import LSTDQSolver, PICESolver


import numpy as np

import pandas as pd
import json
from utils import *
from lspi.policy_ct import convertW2S

from feature_selection import feature_selection

def feature_extractor(data_folder_path, cfg, vis = False, avg_gait_num = -1):
    
    
    params = np.array([file[:-4].split('_')for file in os.listdir(data_folder_path) if file.endswith('.csv')] , dtype=float)


    states_list = []
    
    # for file in os.listdir(data_folder_path):
    for param in params:
        # print(f"collecting feature of param set: {param}")
        file = f'{int(param[0])}_{int(param[1])}_{int(param[2])}.csv'
        if file in os.listdir(data_folder_path):
            
            # action_values = (action_values - mins) / (maxs - mins)
            # print(f"action_values: {action_values}, {action_values_}")
            with open(os.path.join(data_folder_path, file), 'r') as f:
                
                # use pandas to read, first row is variable names, values start from the second row
                data, sw_st_idx, st_sw_idx = extract_gait_data(f,cfg)
                
                 # use mean +- 3* std to skip outliers, return the clean whole data with mean and std of the var
                data_list = skip_outliers( data, sw_st_idx, cfg, "ACTUATOR_POSITION", test = False)
                if len(data_list) == 0:
                    print(f"no data for {file}")
                    # pop the param from params
                    params = np.delete(params, np.where(params == param)[0][0], axis=0)
                    continue

                
                if vis:
                    visualize_gait_cycle(cfg, file, data_list)
                
                if avg_gait_num > 1:
                    # if there are multiple gait cycles, take the moving average of the data
                    data_list = moving_average(data_list, avg_gait_num)
                    print(f"moving average of {len(data_list)} gait cycles with {avg_gait_num} gait cycles")

                state_values, names = feature_selection(data_list)                # print(f"For param set: {param}, state values avg: {state_values.mean(axis=0)}, std: {state_values.std(axis=0)}")
                # Append action values to all state samples and save as npy array
                states_list.append(state_values) # states represent features from single gait cycle
                
    return params, states_list, np.array(names)


def create_ssa_samples(params, states, s_target, n_samples = 1000, normalize = True, n = 4):
    """
    shuffle the states and params, and create samples"""
    
    
    pair_idx = np.random.randint(0, len(states), size = (n_samples, 2))
    ssa_samples= []
    
    for i in range(n_samples):
        # print(f"pair_idx: {pair_idx[i]}")
        state = states[pair_idx[i, 0]]
        next_state = states[pair_idx[i, 1]]
        n_group = min(n, len(state))
        within_param_set_idx = np.random.randint(0, len(state)-n_group + 1) # avg group mean 
        state= np.mean(state[within_param_set_idx: within_param_set_idx + n_group], axis = 0) # calculate mean of a consecutive n_group samples
        n_group = min(n, len(next_state))
        within_param_set_idx = np.random.randint(0, len(next_state)-n_group + 1)
        next_state= np.mean(next_state[within_param_set_idx: within_param_set_idx + n_group], axis = 0)

        action = params[pair_idx[i, 1]] - params[pair_idx[i, 0]]

        ssa_samples.append( np.concatenate([state - s_target, next_state - s_target, action], axis = 0))
    ssa_samples = np.array(ssa_samples)
    # print(f"ssa_samples.shape: {ssa_samples.shape}")
    if normalize:
        # Normalize the samples
        n_s = states[0].shape[1]
        mean_s, std_s = np.mean(ssa_samples[:, :n_s], axis=0), np.std(ssa_samples[:, :n_s], axis=0)
        mean_a, std_a = np.mean(ssa_samples[:, 2* n_s:], axis=0), np.std(ssa_samples[:, 2*n_s:], axis=0)
        # normalize states and action respectively
        ssa_samples[:, :n_s] = (ssa_samples[:, :n_s] - mean_s) / std_s
        ssa_samples[:, n_s:2*n_s] = (ssa_samples[:, n_s:2*n_s] - mean_s) / std_s
        ssa_samples[:, 2*n_s:] = (ssa_samples[:, 2*n_s:] - mean_a) / std_a
        return ssa_samples, [mean_s, mean_a], [std_s, std_a]

    
    return ssa_samples, [], []

def add_quadratic_reward_stack(ssa_samples,  cfg, w_s = 0.8):
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


def save_parameters_and_states(folder_path, params, names, states):
    np.save(os.path.join(folder_path, "params.npy"), params)
    np.save(os.path.join(folder_path, "param_names.npy"), names)
    for i in range(len(states)):
        state = states[i]
        np.save(os.path.join(folder_path, f"state_{i}.npy"), states[i])


def data_loader(folder_path, cfg_path, use_save=False):
    """
    Load and preprocess data for lspi.

    Args:
        folder_path (str): Path to the data folder.
        cfg_path (str): Path to the configuration JSON file.
        use_save (bool): Whether to use saved data or process raw data.

    Returns:
        tuple: Processed parameters, states, and SSA samples with offsets and scales.
    """
    cfg = json.load(open(cfg_path))
    param_names = [v["name"] for v in cfg["control_variables"]]

    if use_save:
        params = np.load(os.path.join(folder_path, "params.npy"))
        states = [np.load(os.path.join(folder_path, f"state_{i}.npy")) for i in range(len(params))]
    else:
        params, states, names = feature_extractor(data_folder_path=folder_path, cfg=cfg, vis = True, avg_gait_num = 4)
        save_parameters_and_states(folder_path, params, names, states) # save the params and states for future use
    # Print stats
    print_state_stats(params, states, names if 'names' in locals() else None)

    return params, states, names if 'names' in locals() else None

def create_ssar(folder_path, params, states, state_target, cfg, normalize=True, save=True):

    # Create SSA samples
    ssa_samples, offset, scale = create_ssa_samples(
        params,
        states,
        s_target= state_target,
        n_samples=1000,
        normalize=normalize,
        n=100
    )
    print("Offset, scale: ", offset, scale)
    # Add quadratic reward stack
    ssar = add_quadratic_reward_stack(np.array(ssa_samples), cfg=cfg, w_s=1.0)
        
    if save:
        np.save(os.path.join(folder_path, "ssar.npy"), ssar)
        np.save(os.path.join(folder_path, "offset_s.npy"), offset[0])
        np.save(os.path.join(folder_path, "offset_a.npy"), offset[1])
        np.save(os.path.join(folder_path, "scale_s.npy"), scale[0])
        np.save(os.path.join(folder_path, "scale_a.npy"), scale[1])

    return ssar, offset, scale

def load_states(folder_path):
    params = np.load(os.path.join(folder_path, "params.npy"))
    state_names = np.load(os.path.join(folder_path, "param_names.npy"))
    ids = []
    for i in range(len(params)):
        if True: #params[i][2] < 70:
            ids.append(i)
    states = []
    for i in ids: #range(len(params)):
        print(f"Loading state {i}, param {params[i]}")
        state = np.load(os.path.join(folder_path, f"state_{i}.npy"))
        states.append(state)

    state_mean = [np.mean(s, axis=0) for s in states]
    state_std = [np.std(s, axis=0) for s in states]
    state_mean, state_std = np.array(state_mean), np.array(state_std)

    params = np.array(params[ids])[:, :3] 
    
    return params, state_names, states,  state_mean, state_std

# def run_emulation(policy, emulator, n_steps=100):
#     """
#     Run the emulator with the given policy for a number of steps.
#     """
#     history = []
#     emulator.reset()
#     for _ in range(n_steps):
#         action = policy.sample_action(emulator.params)
#         next_state = emulator.step(action)
#         history.append(emulator.params.copy())
#         print("param history:", history[-1], "Next state:", next_state)
    
#     return history


if __name__ == "__main__":
    # folder_path = "/Users/xinyi/Documents/Data/ossur/DC_04_26"
    folder_path = "/Users/xinyi/Documents/Data/ossur/DC_05_18"
    cfg_path = f"{folder_path}.json"
    if False: #not os.path.exists(os.path.join(folder_path, "params.npy")):
        print(f"Processing data in {folder_path}...")
        params, states, state_names = data_loader(folder_path, cfg_path, use_save=False) # first time run this to extract features and save them
        state_mean = np.array([np.mean(s, axis=0) for s in states])
        state_std = np.array([np.std(s, axis=0) for s in states])
    else:
        params, state_names, states,  state_mean, state_std = load_states(folder_path)

    print_state_stats(params, states, state_names)

    # # add trip occurance as one other feature manually
    # trip_occurance = np.zeros((len(params),))
    # trip_occurance = np.array([trip_occurance[tuple(param)] for param in params])
    # # add trip occurance as a feature
    # state_mean = np.concatenate([state_mean, trip_occurance.reshape(-1, 1)], axis=1)
    # state_std = np.concatenate([state_std, np.zeros((state_std.shape[0], 1))], axis=1) # std is 0 for trip occurance
    # state_names = np.concatenate([state_names, ["Trip Occurance"]], axis=0) # add trip occurance to state names

    RA = True # whether to do feature regression analysis and rank features
    if RA:
        
        # normalize the feature
        state_mean_normalized = (state_mean - np.mean(state_mean, axis=0)) / np.std(state_mean, axis=0)
        # normalize the predictor
        params_normalized = (params - np.mean(params, axis=0)) / np.std(params, axis=0)
        results_df, model = feature_regression_analysis(state_mean_normalized, 
                                                 state_std, 
                                                 state_names, 
                                                 params_normalized, param_names=["Init STF angle", "SWF target", "SW init"],vis = False)
        
        # 1st order approximation of the dynamics 
        
        states = [states[i][:,results_df.index] for i in range(len(states))]
        # rank based on p-value (results_df index)
        state_names, state_mean, state_std = state_names[results_df.index], state_mean[:, results_df.index], state_std[:,results_df.index]

    # specify by name:
    target_names = ["st_sw_phase", "toe_off_time", "brake_time"]
    n_state = len(state_names)
    # filter states by names
    states = [s[:, [np.where(state_names == name)[0][0] for name in target_names]] for s in states]
    state_mean = np.array([np.mean(s, axis=0) for s in states])
    state_std = np.array([np.std(s, axis=0) for s in states])
    n_action = params.shape[1]
    state_names = np.array(target_names)
    print(f"state_names: {state_names}, mean: {state_mean.mean(axis=0)}")

   
    ## Create SSA samples and save them
    ssar, offset, scale = create_ssar(folder_path, params, states, 
                                      state_target = state_mean.mean(axis=0), 
                                      cfg=json.load(open(cfg_path)), save=True)
    

    print(f"ssar shape: {ssar.shape}, n_state: {n_state}, n_action: {n_action}")
    # exit()
    samples = load_from_data(ssar, n_state, n_action)

    solver = PICESolver()#LSTDQSolver()
    policy, all_policies = lspi_loop_offline(solver, samples, 
                                             discount=0.9,
                                             epsilon=0.001, 
                                             max_iterations=1)
    
    
    # Convert weights to state-action space
    print(convertW2S(policy.weights))
    np.save("policy_weights.npy", policy.weights) 

    

    # check if the policy makes sense
    # create pseudo samples as the group mean of different param sets.
    pseudo_samples = []
    for i in range(1): #len(states)):
        state = states[i].mean(axis=0) -  np.zeros(n_state )
        state = (state - offset[0])/scale[0]
        policy_action = (policy.best_action(state) + offset[1]) * scale[1]
        
        print(f"param set {i}: {params[i]}, state: {state}, action: {policy_action}")
    
    pseudo_samples = np.array(pseudo_samples)

    




    
    
