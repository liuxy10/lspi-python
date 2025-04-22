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
    print(params)
    # for file in os.listdir(data_folder_path):
    for param in params:
        
        file = f'{int(param[0])}_{int(param[1])}_{int(param[2])}.csv'
        if file in os.listdir(data_folder_path):
            
            # action_values = (action_values - mins) / (maxs - mins)
            # print(f"action_values: {action_values}, {action_values_}")
            with open(os.path.join(data_folder_path, file), 'r') as f:
                
                # use pandas to read, first row is variable names, values start from the second row
                data, sw_st_idx = extract_gait_data(f,cfg)
                
                 # use mean +- 3* std to skip outliers
                data_list, mean, std = skip_outliers( data, sw_st_idx, cfg)

                plot_figure = True
                if plot_figure:
                    visualize_gait_cycle(cfg, file, data_list, mean, std)
                
                state_values = np.array([[dat["KNEE_ANGLE"].max(), 
                                          dat["KNEE_ANGLE"].min(),
                                        #   dat["KNEE_ANGLE"].idxmin(), ## TODO: not global max/min index
                                        #   dat["KNEE_ANGLE"].idxmax()
                                          ] for dat in data_list])
                # append action values to all state samples and save as npy array
                states_list.append(state_values)
                
    return params, states_list


def create_ssa_samples(params, states, n_samples = 1000, normalize = True):
    
    pair_idx = np.random.randint(0, len(states), size = (n_samples, 2))
    ssa_samples= []
    for i in range(n_samples):
            # print(f"pair_idx: {pair_idx[i]}")
        state, next_state = states[pair_idx[i, 0]], states[pair_idx[i, 1]]
            
        state, next_state = state[np.random.randint(0, len(state))], next_state[np.random.randint(0, len(next_state))]
        action = params[pair_idx[i, 1]] - params[pair_idx[i, 0]]

        ssa_samples.append( np.concatenate([state, next_state, action], axis = 0))
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
        policy = lspi.learn(samples, initial_policy, solver, epsilon=1e-2)
        # Compute the distance between the. current and the previous policy
        assert len(policy.weights) == len(all_policies[-1].weights), "Policy weights do not match"
        difference = policy.weights - all_policies[-1].weights
        lmax_norm = np.linalg.norm(difference, np.inf)
        l2_norm = np.linalg.norm(difference)
        # else:
        #     lmax_norm = abs(np.linalg.norm(policy.weights, np.inf) -
        #                     np.linalg.norm(all_policies[-1].weights, np.inf))
        #     l2_norm = abs(np.linalg.norm(policy.weights) -
        #                   np.linalg.norm(all_policies[-1].weights))
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
    cfg = json.load(open("/Users/xinyi/Documents/Data/ExperimentData/DC_04_07.json"))
    action_names = [v["name"] for v in cfg["control_variables"]]
    n_action = len(action_names)
    raw_variable_names = ["TIME_SYSTEM_ON", 
                          "ACTIVITY", 
                          "GAIT_SUBPHASE", 
                          "ACTUATOR_POSITION", 
                          "TORQUE_ESTIMATE", 
                          "ACTUATOR_SETPOINT", 
                          "LOADCELL", 
                          "LINEAR_ACC_X_LOCAL", 
                          "LINEAR_ACC_Y_LOCAL", 
                          "LINEAR_ACC_Z_LOCAL", 
                          "GRAVITY_VECTOR_X", 
                          "GRAVITY_VECTOR_Y", 
                          "GRAVITY_VECTOR_Z", 
                          "SHANK_ANGLE"]
    use_save = False
    if use_save:
        ssar = np.load("ssar.npy")
    else:
        params, states = feature_extractor(data_folder_path= "/Users/xinyi/Documents/Data/ExperimentData/DC_04_07",
                          cfg= cfg)
        
        n_state = states[0].shape[1]
        assert len(states) == params.shape[0], "states and params should have the same length"

        ssa_samples = create_ssa_samples(params, states, n_samples= 1000, normalize = True)
        ssar = add_quadratic_reward_stack(np.array(ssa_samples), cfg = cfg,
                                    s_target = np.array([
                                                        #  0.8, # max KA phase var
                                                        #  0.6, # min KA phase var
                                                         0.7, # max KA angle
                                                         -0.2 # min KA angle
                                                         ])) 
        np.save("ssar.npy",ssar) 
        # exit(0)
    # slice based on action values
    # eg. column number -3: 'init_STF_ang' (0 25 50 75 100), -2: 'SWF_target_ang' ([40. 48. 57. 66. 75.]), -1:'TOA' (0-100)
    mins =np.array([v["min"] for v in cfg["control_variables"]])
    maxs = np.array([v["max"] for v in cfg["control_variables"]])


    # slice = (ssar[:, -n_action] == 50) * (ssar [:, -n_action + 1] == 57)
    # ssar = ssar[slice, :-2]
    # # exit(0)
    samples = load_from_data(ssar, n_state, n_action)

    solver = LSTDQSolver()
    policy, all_policies = lspi_loop_offline(solver, samples, discount=0.8, epsilon=0.01, max_iterations=1)
    
    
    # Convert weights to state-action space
    print(convertW2S(policy.weights))



    
    
