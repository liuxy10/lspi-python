import numpy as np
import os
import sys
from lspi.policy import Policy
from lspi.basis_functions import ExactBasis, RadialBasisFunction
from lspi.policy import Policy
from lspi.policy_ct import QuadraticPolicy
from lspi.sample import Sample
from lspi.solvers import LSTDQSolver
import lspi
from lspi import domains
import numpy as np

import pandas as pd
import json
from utils import *
from lspi.policy_ct import convertW2S



def feature_extractor(data_folder_path, cfg):
    
    
    params = np.array([file[:-4].split('_')for file in os.listdir(data_folder_path) if file.endswith('.csv')] , dtype=float)

    mins =np.array([v["min"] for v in cfg["control_variables"]])
    maxs = np.array([v["max"] for v in cfg["control_variables"]])
    print(f"mins: {mins}")
    print(f"maxs: {maxs}")

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

                plot_figure = True
                if plot_figure:
                    visualize_gait_cycle(cfg, file, data_list)
                
                state_values = np.array([[dat["ACTUATOR_POSITION"].max(), 
                                        #   dat["ACTUATOR_POSITION"].min(),
                                        np.argmax(dat["ACTUATOR_POSITION"])/100,
                                        # np.argmin(dat["ACTUATOR_POSITION"])/100
                                          ] for dat in data_list])
                print(f"For param set: {param}, state values avg: {state_values.mean(axis=0)}, std: {state_values.std(axis=0)}")
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
        ssa_samples = (ssa_samples - np.mean(ssa_samples, axis=0)) / np.std(ssa_samples, axis=0)
    

    
    return ssa_samples


def lspi_loop_offline(solver, samples, discount, epsilon, max_iterations = 5, initial_policy=None):

    # Initialize random seed
    # np.random.seed(int(sum(100 * np.random.rand())))
    n_action = samples[0].action.shape[0]
    n_state = samples[0].state.shape[0]
    # Create a new policy
    policy = QuadraticPolicy(n_action= n_action, n_state= n_state, explore = 0.01, discount = discount)
    if initial_policy is None:
        initial_policy = policy.cp()
    
    # Initialize policy iteration
    iteration = 0
    distance = float('inf')
    all_policies = [initial_policy.cp()]
 

    # If no samples, return
    if not samples:
        print('Warning: Empty sample set')
        return policy, all_policies
    # Main LSPI loop
    while iteration < max_iterations and distance > epsilon:
        # Update and print the number of iterations
        iteration += 1
        print('*********************************************************')
        print(f'LSPI iteration: {iteration}')
        iteration == 1
        # Evaluate the current policy (and implicitly improve)
        policy = lspi.learn(samples, initial_policy.cp(), solver, epsilon=1e-2)
        # Compute the distance between the. current and the previous policy
        assert len(policy.weights) == len(all_policies[-1].weights), "Policy weights do not match"
        difference = policy.weights - all_policies[-1].weights
        lmax_norm = np.linalg.norm(difference, np.inf)
        l2_norm = np.linalg.norm(difference)

        distance = l2_norm

        # Print some information
        print(f'Norms -> Lmax: {lmax_norm:.6f}   L2: {l2_norm:.6f}')

        # Store the current policy
        all_policies.append(policy.cp())

    # # Display some info
    # print('*********************************************************')
    # if distance > epsilon:
    #     print(f'LSPI finished in {iteration} iterations WITHOUT CONVERGENCE to a fixed point')
    # else:
    #     print(f'LSPI converged in {iteration} iterations')
    # print('*********************************************************')

    return policy, all_policies



if __name__ == "__main__":
    cfg = json.load(open("/Users/xinyi/Documents/Data/ossur/DC_04_26.json"))
    # cfg = json.load(open("udp/DC_04_26.json")) # for temp use
    action_names = [v["name"] for v in cfg["control_variables"]]
    n_action = len(action_names)
    use_save = False
    if use_save:
        ssar = np.load("ssar.npy")
        n_state = (ssar.shape[1] - n_action - 1) // 2
    else:
        params, states = feature_extractor(
                                        # data_folder_path= "udp/ossur/DC_04_26", 
                                        data_folder_path="/Users/xinyi/Documents/Data/ossur/DC_04_26",
                                        cfg= cfg
                                        )
        
        n_state = states[0].shape[1]
        assert len(states) == params.shape[0], "states and params should have the same length"
        exit(0)
        ssa_samples = create_ssa_samples(params, states, s_target = np.array([
                                                        68, # max KA angle
                                                        #  -0.0, # min KA angle
                                                         0.64, # max KA phase var
                                                        #  0.6 # min KA phase var
                                                         ]), 
                                                        n_samples= 3000, 
                                                        normalize = True,
                                                        n=100) # set n group to make sure every parameter set only have one sample 
        
        ssar = add_quadratic_reward_stack(np.array(ssa_samples), cfg = cfg, w_s= 1.0) # w_s = 1.0, w_a = 0.0
        np.save("ssar.npy",ssar) 
        # exit(0)
    # slice based on action values
    # eg. column number -3: 'init_STF_ang' (0 25 50 75 100), -2: 'SWF_target_ang' ([40. 48. 57. 66. 75.]), -1:'TOA' (0-100)
    # mins =np.array([v["min"] for v in cfg["control_variables"]])
    # maxs = np.array([v["max"] for v in cfg["control_variables"]])


    # slice = (ssar[:, -n_action] == 50) * (ssar [:, -n_action + 1] == 57)
    # ssar = ssar[slice, :-2]
    # # exit(0)
    samples = load_from_data(ssar, n_state, n_action)

    solver = LSTDQSolver()
    policy, all_policies = lspi_loop_offline(solver, samples, discount=0.8, epsilon=0.01, max_iterations=1)
    
    
    # Convert weights to state-action space
    print(convertW2S(policy.weights))
    np.save("policy_weights.npy", policy.weights) 


    



    
    
