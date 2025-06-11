
import numpy as np
from lspi.policy_ct import QuadraticPolicy
from utils import *
import statsmodels.api as sm
from ossur_env_simulator import LRSimulator, Simulator
from gym_env import SimulatorEnv


def vis_Q_value(self, his, skip = 1):
    """ vis Q againt stage cost + gamma * Q_next """
    stage_cost = np.array(his["stage_cost"])[:-1:skip].reshape(-1)  # reshape to 2D array for scatter plot
    Q = np.array(his["q_value"])[:-1:skip].reshape(-1)  # reshape to 2D array for scatter plot
    next_Q = np.array(his["q_value"])[1::skip].reshape(-1)
    Q_next = (stage_cost + self.discount * next_Q).reshape(-1)
    # plot Q values against stage cost + gamma * Q_next and color by index
    plt.figure(figsize=(8, 6))
    plt.scatter(Q, Q_next, c=np.arange(len(Q)), cmap='viridis', s=50, alpha=0.7)
    # add a diagonal line for reference
    q_max = max(Q.max(), Q_next.max())
    plt.xlim(0, q_max)
    plt.ylim(0, q_max)
    plt.plot([0, q_max], [0, q_max], 'r--', linewidth=2, label='y=x')
    plt.colorbar(label='Index')
    plt.xlabel('Q Value')
    plt.ylabel('Stage Cost + Gamma * Q_next')
    plt.title('Q Value vs Stage Cost + Gamma * Q_next')
    # axis equal for better visualization
    plt.axis('equal')
    plt.grid()
    plt.show()

def case_LS_fixed_action():
    """ Example case for the PICE simulator with a simple linear model with 3 inputs and 3 outputs."""
    from lspi.solvers import convertW2S, PICESolver, LSTDQSolver
    from lspi.lspi_train_offline import lspi_loop_offline
    # Example data
    y = np.array([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]]).T  # (n_sample, n_out)
    X = np.array([1, 2, 3, 4, 5])
    # Normalize input and output
    output_offset = np.mean(y, axis=0)
    input_offset = np.mean(X, axis=0)
    output_scale = np.std(y, axis=0)
    input_scale = np.std(X, axis=0)
    y_norm = (y - output_offset) / output_scale
    X_norm = (X - input_offset) / input_scale
    weights = np.ones(len(X))  # Example weights
    model = sm.WLS(y_norm, X_norm, weights=weights).fit()

    # Create the simulator from the model
    simulator = Simulator(model, input_scale=input_scale,
                          input_offset=input_offset,
                          output_scale=output_scale,
                          output_offset=output_offset)
    simulator.reset()

    # Use LSPI and QuadraticPolicy to provide actions

    n_state = 2  # Since X is 1D
    n_action = 1  # We'll use 1D action for this example

    # Generate some random samples for LSPI
    samples = []
    state = simulator.predict()
    for _ in range(20):
        action = np.random.uniform(-1, 1, size=(n_action,))
        simulator.step(action)
        next_state = simulator.predict()
        reward = -np.sum((next_state - 0.5) ** 2)  # Example reward
        samples.append((state.copy(), action.copy(), reward, next_state.copy()))
        state = next_state

    samples_dict = sample_to_dict(samples)

    # Initialize policy and solver
    policy = QuadraticPolicy(n_state=n_state, n_action=n_action, weights=np.zeros((n_state + n_action + 1, n_action)))
    solver = LSTDQSolver()
    policy, _ = lspi_loop_offline(solver, samples_dict, initial_policy=policy, discount=0.9, epsilon=0.001, max_iterations=5, verbose=False)

    # Use the learned policy to provide actions
    simulator.reset()
    state = simulator.predict()
    history = {"params": [], "next_state": [], "action": []}
    for _ in range(10):
        action = policy.best_action(state)
        simulator.step(action)
        next_state = simulator.predict()
        history["params"].append(simulator.params.copy())
        history["next_state"].append(next_state.copy())
        history["action"].append(action.copy())
        print("param history:", history["params"][-1], "Next state:", next_state, "Action:", action)
        state = next_state

    for k, v in history.items():
        history[k] = np.array(v)


    simulator.vis_model(history)
    simulator.vis_cost(np.array([0.5, 0.5]), trace=history)

def case_PICE(vis = True, init_params = np.array([60, 60, 60])):
    """ Example case for the PICE simulator with a simple linear model with 3 inputs and 3 outputs."""
    from lspi.solvers import convertW2S, PICESolver, LSTDQSolver
    from lspi.lspi_train_offline import lspi_loop_offline
    folder_path = "/Users/xinyi/Documents/Data/ossur/DC_05_18"
    # simulator 
    simulator = LRSimulator(folder_path=folder_path)
    # load policy
    w = np.load(os.path.join(folder_path, "policy_weights.npy"))  # load weights from file
    print(np.array2string(convertW2S(w), formatter={'float_kind':lambda x: "%.4f" % x})) # convert state to weights
    
    policy = QuadraticPolicy(n_state=3, n_action=3, weights=w,
                              explore = 0.05, # increase exploration
                              discount = 0.9,
                              folder_path = folder_path) # folder is used to load calculated scale and offset
    
    simulator.reset_params(init_params)  # reset the simulator parameters
    state = simulator.predict()
    print("initial params", simulator.params, "initial state", state)
    his = {"current_state": [],  # history of current state
           "next_state": [], # history of next state
           "params": [],  # history of parameters
           "action": [],  # history of actions
           "q_value":[],  # history of next state and parameters
           "stage_cost": [], # history of stage cost
           "param_at_bound": [],
           "start":[]} # indicates if the parameters are at the bound
    n_rl = 10 # number of RL trials per policy update iteration
    n_step = 4 # # number of steps per RL trial
    # External loop for policy update iterations
    for k in range(n_rl):  # for each policy update iteration
        
        print(f"Policy update iteration {k+1}") 
        for i in range(n_step):  # for each trial 

            action = policy.best_action(state)
            simulator.step(action)
            next_state = simulator.predict()

            # print(f"Sample {i}: normState = {state}, normNextState = {next_state}, action = {action}, q_value = {policy.calc_q_value(state, action)}")
            his["current_state"].append(state.copy())
            his["next_state"].append(next_state.copy())
            his["action"].append(action.copy())
            his["params"].append(simulator.get_params())  # record the parameters
            his["q_value"].append(policy.calc_q_value(state, action).copy())
            his["stage_cost"].append(simulator.calc_stage_cost(next_state, target = simulator.output_offset))  # calculate stage cost
            his["param_at_bound"].append(simulator.at_bound()>2)  # check if the parameters are at the bound
            his["start"].append(1 if i == 0 else 0)  # indicate if the parameters are at the bound
            state = next_state

        # Visualize after each policy update
        target = simulator.output_offset #np.array([0., 0., 0.])  # target state for visualization

        # update policy 
        ssa = np.concatenate([his["current_state"].copy() - target, his["next_state"].copy() - target, his["action"].copy()], axis=1)
        ssar = add_quadratic_cost_stack(ssa, simulator.n_action, w_s = 1)
        samples = load_from_data(ssar, simulator.n_state, simulator.n_action)
        solver = LSTDQSolver()
        policy, all_policies = lspi_loop_offline(solver, samples, initial_policy=policy,
                                                    discount=0.9,
                                                    epsilon=0.001, 
                                                    max_iterations=1,
                                                    verbose=False)
        # print(f"Updated policy weights: {np.array2string(convertW2S(policy.weights), formatter={'float_kind':lambda x: '%.4f' % x})}")
        if vis and k % 5 == 0:  # visualize every 5 iterations
            print(f"Visualizing policy at iteration {k}")
            # Clear previous plots to update in each iteration
            plt.close('all')
            # Plot cost trace
            simulator.vis_cost(target, trace=his)
            # Plot parameter/state/action history
            simulator.vis_hist(his, target=target)
            # Visualize Q values
            vis_Q_value(policy, his, skip=1)
            plt.pause(0.1)  # Pause to allow the plot to update


        # exit condition: minimum stage cost
        if k > 0 and np.max(np.abs(np.array(his["next_state"][-5:]) - target)) < simulator.param_region:  # check if the policy has converged
            print(f"At i = {k}, global minimum reached, stopping training.")
            return True, his
        # reset conditions:
        elif k > 0 and np.mean(his["param_at_bound"][-5:]) > 0.8:
            print(f"At i = {k}, Parameters at bound, resetting simulator.")
            simulator.reset_to_hist_best_or_random(init_params,  his, his_window = 10)
            state = simulator.predict()
               
        elif k>0 and np.linalg.norm(policy.weights - all_policies[-2].weights) < 0.1:
            print(f"At i = {k},convergence to a local minimum, not global, at p = 0.5, simulator reset.")
            if np.random.random()< 0.5:
                print("resetting")
                simulator.reset_to_hist_best_or_random(init_params,  his, p_init = 0.1, p_random = 0.2, p_pass = 0.5)
                
                # state = simulator.predict()
                # print(f"Resetting to historic best params {simulator.params} with stage cost {np.mean(his['stage_cost'][-5:]):.4f}")
            # simulator.reset_to_hist_best_or_random(init_params,  his)
            
        
    return False, his 

def case_DDPG(vis=True):
    from ddpg_agent import DDPGAgent
    import torch
    folder_path = "/Users/xinyi/Documents/Data/ossur/DC_05_18"
    simulator = LRSimulator(folder_path=folder_path)
    agent = DDPGAgent(state_dim=3, action_dim=3)
    
    simulator.reset_params(np.array([60, 60, 60]))
    state = simulator.predict()
    
    history = {"states": [], "actions": [], "rewards": [], "params": []}
    target = np.array([0.5, 0.5, 0.5])  # target state for visualization
    for episode in range(1000):
        action = agent.get_action(state)
        
        # Take action in environment
        simulator.step(action)
        next_state = simulator.predict()
        
        # Calculate reward (customize based on your task)
        reward = -np.sum((next_state - target))**2
        
        # Store transition
        done = False  # Set your termination condition
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # Update agent
        agent.update()
        
        # Record history
        history["states"].append(state.copy())
        history["actions"].append(action.copy())
        history["params"].append(simulator.params.copy())
        history["rewards"].append(reward)
        
        state = next_state
        
        if episode % 30 == 0:
            print(f"Episode {episode}, Reward: {np.mean(history['rewards'][-10:]):.2f}")
        # clean cache
        torch.cuda.empty_cache()
    
    
    if vis:
        simulator.vis_cost(target, trace=history, skip = 50)
        simulator.vis_hist(history, skip=50, target=target)  # visualize the history of the policy


def case_PPO_gym():
    # Example usage with Stable Baselines3
    from stable_baselines3 import PPO

    # Initialize environment
    from lspi.solvers import convertW2S, PICESolver, LSTDQSolver
    from lspi.lspi_train_offline import lspi_loop_offline
    folder_path = "/Users/xinyi/Documents/Data/ossur/DC_05_18"
    # simulator 
    simulator = LRSimulator(folder_path=folder_path)


    target = simulator.output_offset   # Your target state
    env = SimulatorEnv(simulator, target, max_steps=10000, cost_threshold = 0.02)

    # Create and train agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1e5)

    
    # Test trained agent
    obs = env.reset()
    env.history["current_state"].append(obs.copy())
    for i in range(1000):
        action, _states = model.predict(obs)
        next_obs, rewards, done, info = env.step(action)
              # Append to history
        env.history["next_state"].append(next_obs.copy())
        env.history["params"].append(info['params'])
        env.history["action"].append(info['action'])
        env.history["stage_cost"].append(info['stage_cost'])
        env.history["param_at_bound"].append(info['param_at_bound'])
        env.history["start"].append(1 if i == 0 else 0)
        
        # Update current state for next iteration
        env.history["current_state"].append(next_obs.copy())
        print(f"Step {i}, Action: {action}, Reward: {rewards}, Next State: {next_obs}")
            
        if i % 30 == 29:
            env.render()
        if done:
            print(f"Episode finished after {i+1} steps.")
            env.render()
            obs = env.reset()
            env.history["current_state"].append(obs.copy())
            break
        else:
            obs = next_obs
    env.close()

def eval_case(func, n_eval,  vis = False, **kwargs):
    from tqdm import tqdm
    from lspi.lspi_train_offline import lspi_loop_offline
    from lspi.solvers import LSTDQSolver
    """ Evaluate a case function with randomized init params """
    bounds = np.array([[40., 75.],[0., 100.],[0., 100.]])  # bounds for the parameters, normalized
    results = {"success": [], 
               "final stage cost": [], 
               "cost_reduction_ratio": [],
               "init_params": [],
               "stage_cost":[]}  # store success results
    for i in tqdm(range(n_eval), desc="Evaluations"):
        init_params = np.random.uniform(bounds[:, 0], bounds[:, 1])
        print(f"Evaluation {i+1}/{n_eval}, init_params: {init_params}")
        success, his = func(vis=vis, init_params=init_params, **kwargs)
        results["init_params"].append(init_params)
        results["success"].append(success)
        results["stage_cost"].append(np.mean(his["stage_cost"][-5:]))
        results["cost_reduction_ratio"].append(np.mean(his["stage_cost"][-5:]) / np.mean(his["stage_cost"][:5]))  # cost reduction ratio

    
    success_rate = np.mean(results["success"])
    print(f"Success rate: {success_rate:.2f} ({np.sum(results['success'])}/{n_eval})")
    print(f"Average final stage cost: {np.mean(results['final stage cost']):.4f}")
    print(f"Average cost reduction ratio: {np.mean(results['cost_reduction_ratio']):.4f}")

    for k, v in results.items():
        if k not in ["init_params", "success"]:
            results[k] = np.array(v)
            # if v is not true or false, visualize the histogram
            if not np.all(np.isin(v, [True, False])):
                plt.figure(figsize=(8, 4))
                plt.hist(v, bins=50, alpha=0.7)
                plt.title(f"Histogram of {k}")
                plt.xlabel(k)
                plt.ylabel("Frequency")
                plt.grid()
                plt.show()
    
  
 

if __name__ == "__main__":

    # case_LS_fixed_action()

    # case_PICE(vis = True)
    # case_PICE_matlab(vis = True)  # evaluate the PICE case with MATLAB LQR Policy Iteration
    # eval_case(case_PICE, n_eval=100, vis = False)  # evaluate the PICE case with 10 evaluations
    
    # case_DDPG()
    case_PPO_gym()  # evaluate the PPO case with gym environment

   

    
