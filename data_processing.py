import numpy as np
import os

from lspi.train_offline import lspi_loop_offline
from lspi.solvers import LSTDQSolver, PICESolver


import numpy as np

import pandas as pd
import json
from utils import *
from lspi.policy_ct import convertW2S



def feature_extractor(data_folder_path, cfg):
    
    
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
                # print(sw_st_idx)
                

                 # use mean +- 3* std to skip outliers, return the clean whole data with mean and std of the var
                data_list = skip_outliers( data, sw_st_idx, cfg, "ACTUATOR_POSITION", test = False)
                if len(data_list) == 0:
                    print(f"no data for {file}")
                    # pop the param from params
                    params = np.delete(params, np.where(params == param)[0][0], axis=0)
                    continue
                
                

                plot_figure = False
                if plot_figure:
                    visualize_gait_cycle(cfg, file, data_list)
                
                state_values = np.array([[dat["ACTUATOR_POSITION"].max(), 
                                        #   dat["ACTUATOR_POSITION"].min(),
                                        np.argmax(dat["ACTUATOR_POSITION"])/100,
                                        # np.argmin(dat["ACTUATOR_POSITION"])/100
                                          ] for dat in data_list])
                # print(f"For param set: {param}, state values avg: {state_values.mean(axis=0)}, std: {state_values.std(axis=0)}")
                # Append action values to all state samples and save as npy array
                states_list.append(state_values) # states represent features from single gait cycle
                
    return params, states_list


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


def save_parameters_and_states(folder_path, params, states):
    np.save(os.path.join(folder_path, "params.npy"), params)
    for i in range(len(states)):
        state = states[i]
        np.save(os.path.join(folder_path, f"state_{i}.npy"), states[i])


def data_loader(folder_path, cfg_path, use_save=False):
    """
    Load and preprocess data for LSPI.

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
        params, states = feature_extractor(data_folder_path=folder_path, cfg=cfg)
        save_parameters_and_states(folder_path, params, states)

    # Slice based on params
    slice_params = np.array([63, -1, 41])
    slice_id = slice_params > 0
    print("Slicing params based on: ", slice_params)
    mask = np.all(params[:, slice_id] == slice_params[slice_id], axis=1)
    params = params[mask][:, ~slice_id]
    states = [states[i] for i in range(len(states)) if mask[i]]

    # Print stats
    for i in range(len(states)):
        state = states[i]
        print(f"Param set {i}: {params[i]} with {state.shape[0]} samples")
        print(f"State avg, std: {state.mean(axis=0)}, {state.std(axis=0)}")
        print("*" * 20)

    # Create SSA samples
    ssa_samples, offset, scale = create_ssa_samples(
        params,
        states,
        s_target=np.array([60, 0.66]),
        n_samples=1000,
        normalize=True,
        n=100
    )
    print("Offset, scale: ", offset, scale)

    # Add quadratic reward stack
    ssar = add_quadratic_reward_stack(np.array(ssa_samples), cfg=cfg, w_s=1.0)
    np.save(os.path.join(folder_path, "ssar.npy"), ssar)
    np.save(os.path.join(folder_path, "offset_s.npy"), offset[0])
    np.save(os.path.join(folder_path, "offset_a.npy"), offset[1])
    np.save(os.path.join(folder_path, "scale_s.npy"), scale[0])
    np.save(os.path.join(folder_path, "scale_a.npy"), scale[1])

    return params, states, ssar, offset, scale


if __name__ == "__main__":
    folder_path = "/Users/xinyi/Documents/Data/ossur/DC_04_26"
    cfg_path = f"{folder_path}.json"
    params, states, ssar, offset, scale = data_loader(folder_path, cfg_path, use_save=False)
    # exit(0)

    n_action = params.shape[1]
    n_state = states[0].shape[1]
    samples = load_from_data(ssar, n_state, n_action)

    solver = PICESolver()#LSTDQSolver()
    policy, all_policies = lspi_loop_offline(solver, samples, discount=0.8, epsilon=0.01, max_iterations=1)
    
    
    # Convert weights to state-action space
    print(convertW2S(policy.weights))
    np.save("policy_weights.npy", policy.weights) 

    # check if the policy makes sense
    # create pseudo samples as the group mean of different param sets.
    pseudo_samples = []
    for i in range(len(states)):
        state = states[i].mean(axis=0) - np.array([60, 0.66])
        state = (state - offset[0])/scale[0]
        policy_action = (policy.best_action(state) + offset[1]) * scale[1]
        
        print(f"param set {i}: {params[i]}, state: {state}, action: {policy_action}")
    
    pseudo_samples = np.array(pseudo_samples)

    



    



    
    
