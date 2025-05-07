import numpy as np
import os
import json
import pandas as pd
from lspi.policy_ct import QuadraticPolicy, convertW2S
from lspi.solvers import LSTDQSolver
import lspi
from utils import extract_gait_data, skip_outliers, visualize_gait_cycle, load_from_data

class OfflineTrainer:
    def __init__(self, folder_path, config_path=None):
        self.folder_path = folder_path
        self.cfg = json.load(open(config_path or f"{folder_path}.json"))
        self.param_names = [v["name"] for v in self.cfg["control_variables"]]
        self.params = None
        self.states = None
        self.n_state = None
        self.n_action = len(self.param_names)
        self.offset = None
        self.scale = None
        self.ssar = None
        self.samples = None
        self.policy = None
        self.all_policies = None

    def feature_extractor(self):
        data_folder_path = self.folder_path
        cfg = self.cfg
        self.params = np.array([file[:-4].split('_') for file in os.listdir(data_folder_path) if file.endswith('.csv')], dtype=float)
        states_list = []
        for param in self.params:
            file = f'{int(param[0])}_{int(param[1])}_{int(param[2])}.csv'
            if file in os.listdir(data_folder_path):
                with open(os.path.join(data_folder_path, file), 'r') as f:
                    data, sw_st_idx, st_sw_idx = extract_gait_data(f, cfg)
                    data_list = skip_outliers(data, sw_st_idx, cfg, "ACTUATOR_POSITION", test=False)
                    if len(data_list) == 0:
                        print(f"Warning: No data for {file}")
                        self.params = np.delete(self.params, np.where(self.params == param)[0][0], axis=0)
                        continue
                    plot_figure = True
                    if plot_figure:
                        visualize_gait_cycle(cfg, file, data_list)
                    state_values = np.array([
                        [dat["ACTUATOR_POSITION"].max(),
                         np.argmax(dat["ACTUATOR_POSITION"]) / 100]
                        for dat in data_list])
                    states_list.append(state_values)
        self.params = self.params
        self.states = states_list
        self.n_state = states_list[0].shape[1]
        return self.params, states_list

    def create_ssa_samples(self, s_target, n_samples=1000, normalize=True, n=4):
        params, states = self.params, self.states
        pair_idx = np.random.randint(0, len(states), size=(n_samples, 2))
        ssa_samples = []
        for i in range(n_samples):
            state = states[pair_idx[i, 0]]
            next_state = states[pair_idx[i, 1]]
            n_group = min(n, len(state))
            within_param_set_idx = np.random.randint(0, len(state) - n_group + 1)
            state = np.mean(state[within_param_set_idx:within_param_set_idx + n_group], axis=0)
            n_group = min(n, len(next_state))
            within_param_set_idx = np.random.randint(0, len(next_state) - n_group + 1)
            next_state = np.mean(next_state[within_param_set_idx:within_param_set_idx + n_group], axis=0)
            action = params[pair_idx[i, 1]] - params[pair_idx[i, 0]]
            ssa_samples.append(np.concatenate([state - s_target, next_state - s_target, action], axis=0))
        ssa_samples = np.array(ssa_samples)
        if normalize:
            n_s = states[0].shape[1]
            mean_s, std_s = np.mean(ssa_samples[:, :n_s], axis=0), np.std(ssa_samples[:, :n_s], axis=0)
            mean_a, std_a = np.mean(ssa_samples[:, 2 * n_s:], axis=0), np.std(ssa_samples[:, 2 * n_s:], axis=0)
            ssa_samples[:, :n_s] = (ssa_samples[:, :n_s] - mean_s) / std_s
            ssa_samples[:, n_s:2 * n_s] = (ssa_samples[:, n_s:2 * n_s] - mean_s) / std_s
            ssa_samples[:, 2 * n_s:] = (ssa_samples[:, 2 * n_s:] - mean_a) / std_a
            self.offset = [mean_s, mean_a]
            self.scale = [std_s, std_a]
        self.ssar = ssa_samples
        return ssa_samples

    def add_quadratic_reward_stack(self, w_s=0.8):
        ssa_samples = self.ssar
        cfg = self.cfg
        n_total = ssa_samples.shape[1]
        s_err = ssa_samples[:, :self.n_state] 
        w_a = np.sqrt(1 - w_s ** 2)
        rew = -0.5 * np.sum(s_err ** 2, axis=1) * w_s - 0.5 * np.sum(ssa_samples[:, - self.n_action:] ** 2, axis=1) * w_a
        rew = rew.reshape(-1, 1)
        self.ssar = np.concatenate([ssa_samples, rew], axis=1)
        return self.ssar

    def save_data(self):
        np.save(os.path.join(self.folder_path, "params.npy"), self.params)
        for i, state in enumerate(self.states):
            np.save(os.path.join(self.folder_path, f"state_{i}.npy"), state)
        np.save(os.path.join(self.folder_path,"ssar.npy"), self.ssar)
        if self.offset and self.scale:
            np.save(os.path.join(self.folder_path,"offset_s.npy"), self.offset[0])
            np.save(os.path.join(self.folder_path,"offset_a.npy"), self.offset[1])
            np.save(os.path.join(self.folder_path,"scale_s.npy"), self.scale[0])
            np.save(os.path.join(self.folder_path,"scale_a.npy"), self.scale[1])

    def load_samples(self):
        self.ssar = np.load(os.path.join(self.folder_path,"ssar.npy"))
        self.params = np.load(os.path.join(self.folder_path,"params.npy"))
        self.offset = [np.load(os.path.join(self.folder_path,"offset_s.npy")),
                        np.load(os.path.join(self.folder_path,"offset_a.npy"))]
        self.scale = [np.load(os.path.join(self.folder_path,"scale_s.npy")), 
                      np.load(os.path.join(self.folder_path,"scale_a.npy"))]
        self.states = [np.load(os.path.join(self.folder_path, f"state_{i}.npy")) for i in range(len(self.params))]
        self.n_state = self.states[0].shape[1]
        self.n_action = self.params.shape[1]
        self.samples = load_from_data(self.ssar, self.n_state, self.n_action)

    def lspi_loop_offline(self, discount=0.8, epsilon=0.01, max_iterations=5):
        solver = LSTDQSolver()
        samples = self.samples
        n_action = samples[0].action.shape[0]
        n_state = samples[0].state.shape[0]
        policy = QuadraticPolicy(n_action=n_action, n_state=n_state, explore=0.01, discount=discount)
        initial_policy = policy.cp()
        iteration = 0
        distance = float('inf')
        all_policies = [initial_policy.cp()]
        if not samples:
            print('Warning: Empty sample set')
            self.policy, self.all_policies = policy, all_policies
            return policy, all_policies
        while iteration < max_iterations and distance > epsilon:
            iteration += 1
            print(f'LSPI iteration: {iteration}')
            policy = lspi.learn(samples, initial_policy.cp(), solver, epsilon=1e-2)
            difference = policy.weights - all_policies[-1].weights
            l2_norm = np.linalg.norm(difference)
            distance = l2_norm
            all_policies.append(policy.cp())
        self.policy, self.all_policies = policy, all_policies
        return policy, all_policies

    def evaluate_policy(self, s_target):
        pseudo_samples = []
        for i in range(len(self.states)):
            state = self.states[i].mean(axis=0) - s_target
            state = (state - self.offset[0]) / self.scale[0]
            policy_action = (self.policy.best_action(state) + self.offset[1]) * self.scale[1]
            print(f"param set {i}: {self.params[i]}, state: {state}, action: {policy_action}")
            pseudo_samples.append(policy_action)
        return np.array(pseudo_samples)
    
    def check(self):    
        # assert np.all(self.ssar == np.load("ssar.npy"))
        assert np.all(self.params == np.load("params.npy"))
        assert len(self.states) == len(self.params)
        assert np.all(self.offset[0] == np.load("offset_s.npy"))
        assert np.all(self.offset[1] == np.load("offset_a.npy"))
        assert np.all(self.scale[0] == np.load("scale_s.npy"))
        assert np.all(self.scale[1] == np.load("scale_a.npy"))


    def slice_action(self, slice_params):
        slice_id = slice_params > 0
        print("slicing params based on: ", slice_params)
        mask = np.all(self.params[:, slice_id] == slice_params[slice_id], axis=1)
        self.params = self.params[mask][:, ~slice_id]
        
        self.states = [self.states[i] for i in range(len(self.states)) if mask[i]]
        self.n_state, self.n_action = self.states[0].shape[1], self.params.shape[1]



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

   


# Example usage:
if __name__ == "__main__":
    folder_path = "/Users/xinyi/Documents/Data/ossur/DC_04_26"
    trainer = OfflineTrainer(folder_path)
    

    use_save = False
    if use_save:
        # ssar = np.load("ssar.npy")
        # n_state = (ssar.shape[1] - n_action - 1) // 2
        trainer.params = np.load(os.path.join(folder_path, "params.npy"))
        trainer.states = []
        for i in range(len(trainer.params)):
            state = np.load(os.path.join(folder_path, f"state_{i}.npy"))
            trainer.states.append(state)
        trainer.load_samples()

    else:
        trainer.feature_extractor()
        

        s_target = np.array([60, 0.66])
        trainer.slice_action(slice_params=np.array([63, -1, 41]))
        trainer.save_data()
        trainer.create_ssa_samples(s_target=s_target, n_samples=1000, normalize=True, n=100)
        trainer.add_quadratic_reward_stack( w_s=0.8)


    # trainer.check()
    

    # trainer.lspi_loop_offline(discount=0.8, epsilon=0.01, max_iterations=1)
    # print(convertW2S(trainer.policy.weights))
    # np.save("policy_weights.npy", trainer.policy.weights)
    # trainer.evaluate_policy(s_target)
    solver = LSTDQSolver()
    policy, all_policies = lspi_loop_offline(solver,trainer.samples, discount=0.8, epsilon=0.01, max_iterations=1)
    
    
    # Convert weights to state-action space
    print(convertW2S(policy.weights))
    np.save("policy_weights.npy", policy.weights) 

    # check if the policy makes sense
    # create pseudo samples as the group mean of different param sets.
    pseudo_samples = []
    for i in range(len(trainer.states)):
        state = trainer.states[i].mean(axis=0) - np.array([60, 0.66])
        state = (state - trainer.offset[0])/trainer.scale[0]
        policy_action = (policy.best_action(state) + trainer.offset[1]) * trainer.scale[1]
        
        print(f"param set {i}: {trainer.params[i]}, state: {state}, action: {policy_action}")
    
    pseudo_samples = np.array(pseudo_samples)
