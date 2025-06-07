
import numpy as np
from lspi.policy_ct import QuadraticPolicy
from data_processing_offline import load_states
from utils import *
import statsmodels.api as sm

class Simulator:
    """ use sm model as a simulator to generate samples
    model: sm.WLS weighted least squares model
    e_out: error in output
    e_in: error in input
    n_action: number of actions
    n_state: number of states
    params: parameters of the simulator, shape (n_action, )
    """

    def __init__(self, model, e_out = 0.01, e_in = 0.01, 
                 input_scale = 1., input_offset = 0, 
                 output_scale = 1., output_offset = 0,
                 stage_cost_weights = None,
                 bound = np.array([[40., 75.],[0., 100.],[0., 100.]])  # bounds for the parameters, normalized
        ):
        self.model = model
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.input_offset = input_offset
        self.output_offset = output_offset
        self.e_out = e_out
        self.e_in = e_in
        # self.n_action = model.model.exog.shape[1]  #number of parameters in the model
        # self.n_state = model.model.endog.shape[1] # number of output features in the model
        self.param_names = np.array([n for n in model.model.exog_names if n != "const"]) # parameter names
        self.n_action = len(self.param_names)  # number of parameters in the model
        self.n_state = input_scale.shape[0] if type(input_scale) is not np.float64 else model.model.endog.shape[1] # number of output features in the model
        self.params = np.zeros(self.n_action) 
        self.weights = stage_cost_weights/np.linalg.norm(stage_cost_weights) if stage_cost_weights is not None else np.ones(self.n_state)/self.n_state  # weights for stage cost function
        self.bound = bound  # bounds for the parameters, normalized
        self.summary()



    def reset(self):
        """ reset the simulator """
        self.params = np.zeros(self.n_action) # change

    def reset_params(self, params):
        assert params.shape == (self.n_action,), "Params shape must match the number of actions"
        """ set the parameters of the simulator """
        self.params = (params - self.input_offset) / self.input_scale  # normalize the parameters
    

    def reset_to_hist_best_or_random(self, init_params, his, p_init = 0.1, p_random = 0.2, p_pass = 0.5, his_window = None):
        p = np.random.uniform(0,1)
        if p < p_init:  # reset to initial params
            print("Resetting to initial params:", init_params)
            self.reset_params(init_params)
        elif p < p_init + p_random:
            print(f"randomly resetting to a new params")
            self.reset_params(np.random.uniform(self.bound[:, 0], self.bound[:, 1]))

        elif p > 1- p_pass: # randomly reset to a new params
            print(f"pass")
            pass
        else: # reset to the best params in history window
            
            if his_window is None: 
                min_stage_cost_index = np.argmin(his["stage_cost"])
                best_params = his["params"][min_stage_cost_index]
            else:
                min_stage_cost_index = np.argmin(his["stage_cost"][-his_window:])
                best_params = his["params"][-his_window:][min_stage_cost_index]
            self.reset_params(best_params)
            print(f"resetting to historic best params {best_params} with stage cost {his['stage_cost'][min_stage_cost_index]:.4f}") # return False if the training did not converge



    def step(self, action):
        assert action.shape == (self.n_action,), "Action shape must match the number of actions"
        """ predict the next state and reward given the action """
        # print("Action:", action, "self.input_scale:", self.input_scale)#, "self.input_offset:", self.input_offset)
        self.params += action / self.input_scale
        for i in range(self.n_action):
            bound = (self.bound - self.input_offset[:, None] )/ self.input_scale[:, None]
        
            if self.params[i] < bound[i, 0]:
                self.params[i] = bound[i, 0]
            elif self.params[i] > bound[i, 1]:
                self.params[i] = bound[i, 1]
    
    def at_bound(self, margin = 0.01):
        """ check if the parameters are at the bound """
        i = 0
        for i in range(self.n_action):
            if self.params[i] < (self.bound[i, 0] - self.input_offset[i]) / self.input_scale[i] + margin or \
               self.params[i] > (self.bound[i, 1] - self.input_offset[i]) / self.input_scale[i] - margin:
                i+= 1
        return i
    
    def _calc_stage_cost(self, y, target):
        return 1/2 * np.inner(self.weights, (y - target)**2)
    def calc_stage_cost(self, y, target):
        y_normalized = (y - self.output_offset) / self.output_scale  # normalize the output
        target_normalized = (target - self.output_offset) / self.output_scale
        return self._calc_stage_cost(y_normalized, target_normalized) 
    
    def predict(self):
        """ predict the next state and reward given the current parameters (scaled and offset) """
        # Add constant term for prediction if needed
        params_with_noise = self.params + self.e_in * np.random.randn(self.n_action)
        if "const" in self.model.model.exog_names:
            params_with_noise = np.concatenate(([1.0], params_with_noise))
        pred = self.model.predict(params_with_noise)
        return self.e_out * np.random.randn(self.n_state) + pred * self.output_scale + self.output_offset
    
    def predict_state_given_param(self, params):
        """ predict the next state given the parameters (scaled and offset) """
        # Add constant term for prediction if needed
        params_with_noise = params + self.e_in * np.random.randn(self.n_action)
        if "const" in self.model.model.exog_names:
            params_with_noise = np.concatenate(([1.0], params_with_noise))
        pred = self.model.predict(params_with_noise)
        return self.e_out * np.random.randn(self.n_state) + pred * self.output_scale + self.output_offset
    
    def get_params(self):
        """ get the current parameters of the simulator """
        return self.params.copy() * self.input_scale + self.input_offset
    
    def vis_model(self, trace = None, n_grid = 20):
        """ visualize the model in normalized space
            plot each feature againt the parameters
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
        
        # sweep each parameter
        if self.n_action == 1:
            params_sweep = np.linspace(-2,2, n_grid) # sweep by n_grid
            # predict features
            y = np.array([self.predict_state_given_param(p).reshape(-1) for p in params_sweep])
            for i in range(y.shape[1]): # plot each feature
                plt.plot(params_sweep, y[:,i], label=f'Feature {i+1}')
                plt.scatter(params_sweep, y[:,i], s=10, alpha=0.5)
            plt.xlabel('X')
            plt.ylabel('y')
            
        elif self.n_action == 2:
            # build meshgrid for 2D parameters
            x = np.linspace(-2, 2, n_grid)
            y = np.linspace(-2, 2, n_grid)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[self.predict_state_given_param(np.array([x_val, y_val])).reshape(-1) for x_val in x] for y_val in y])
            for i in range(Z.shape[2]):  # plot each feature
                plt.figure(figsize=(8, 6))
                plt.contourf(X, Y, Z[:, :, i], levels=20, cmap='viridis', alpha=0.7)
                plt.colorbar(label=f'Feature {i+1}')
                plt.xlabel(self.param_names[0] if self.param_names is not None else 'Parameter 1')
                plt.ylabel(self.param_names[1] if self.param_names is not None else 'Parameter 2')
                plt.title(f'Feature {i+1} vs Parameters')
            
        elif self.n_action == 3:
            # build meshgrid for 3D parameters
            x = np.linspace(-2, 2, 20)
            y = np.linspace(-2, 2, 20)
            z = np.linspace(-2, 2, 20)
            X, Y, Z = np.meshgrid(x, y, z)
            W = np.array([self.predict_state_given_param(np.array([x_val, y_val, z_val])).reshape(-1) for x_val, y_val, z_val in zip(X.flatten(), Y.flatten(), Z.flatten())])
            W = W.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
            
            
            for i in range(W.shape[3]):
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=W[:, :, :, i].flatten(), cmap='viridis', marker='o', alpha=0.2)
                fig.colorbar(ax.collections[0], ax=ax, label=f'Feature {i+1}', shrink = 0.4, aspect=5)
                """ add trace to the figure """
                if trace is not None: 
                    # Convert history to a DataFrame for easier plotting
                    h = np.array(trace).T
                    # Add a trace for each parameter using matplotlib
                    ax.plot(h[0], h[1], h[2], marker = ".", color='black', label='Trace') 
                    ax.plot(h[0,-1], h[1,-1], h[2,-1],marker='*', color='black', label='Trace end') 
                    # mark the end parameter with text
                    ax.text(h[0,-1], h[1,-1], h[2,-1], f'({h[0,-1]:.2f}, {h[1,-1]:.2f}, {h[2,-1]:.2f})', color='black')
                ax.set_xlabel(self.param_names[0] if self.param_names is not None else 'Parameter 1')
                ax.set_ylabel(self.param_names[1] if self.param_names is not None else 'Parameter 2')
                ax.set_zlabel(self.param_names[2] if self.param_names is not None else 'Parameter 3')
                ax.set_title(f'Feature {i+1} vs Parameters')

        else:
            raise ValueError("Model parameters should be 1, 2 or 3 dimensional for visualization.")
        
        plt.title('Simulator Model Visualization')
        plt.legend()
        plt.show()

    def vis_cost(self, target, weights = None, trace=None, n_grid = 20, skip = 1):
        """ visualize the quadratic cost function in normalized space
            plot each feature against the parameters
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
        assert target.shape == (self.n_state,), "Target shape must match the number of states"
        if weights is not None:
            assert weights.shape == (self.n_state,), "Weights shape must match the number of states"

        w = weights if weights is not None else np.ones(self.n_state)  # use weights for cost function, default to 1 for each feature
        # sweep each parameter
        if self.n_action == 1:
            params_sweep = np.linspace(-2,2, n_grid) # sweep by n_grid
            # predict features
            y = np.array([self.predict_state_given_param(p).reshape(-1) for p in params_sweep]) # predict the features
            cost = 1/2 * np.inner(w, (y - target)**2)  # calculate the stage cost
            
            if trace is not None: 
                x_trace = trace["params"].reshape(-1)
                x_cost = np.array([1/2 * np.inner(w, (self.predict_state_given_param(p).reshape(-1) - target)**2) for p in x_trace])
                plt.plot(x_trace[::skip], x_cost[::skip], marker='o', color='black', label='Trace')
                # mark the end parameter with text
                plt.text(x_trace[-1], x_cost[-1], f'({x_trace[-1]:.2f}, {x_cost[-1]:.2f})', color='black')

            plt.plot(params_sweep, cost, label=f'cost')
            plt.scatter(params_sweep, cost, s=10, alpha=0.5)
            plt.xlabel('X')
            plt.ylabel('y')
            
        elif self.n_action == 2:
            # build meshgrid for 2D parameters
            x = np.linspace(-2, 2, n_grid)
            y = np.linspace(-2, 2, n_grid)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[self.predict_state_given_param(np.array([x_val, y_val])).reshape(-1) for x_val in x] for y_val in y])
            for i in range(Z.shape[2]):  # plot each feature
                plt.figure(figsize=(8, 6))
                plt.contourf(X, Y, Z[:, :, i], levels=20, cmap='viridis', alpha=0.7)
                plt.colorbar(label=f'Feature {i+1}')
                plt.xlabel(self.param_names[0] if self.param_names is not None else 'Parameter 1')
                plt.ylabel(self.param_names[1] if self.param_names is not None else 'Parameter 2')
                plt.title(f'Feature {i+1} vs Parameters')
            
        elif self.n_action == 3:
            # build meshgrid for 3D parameters
            x = np.linspace(self.bound[0][0], self.bound[0][1], 20) 
            y = np.linspace(self.bound[1][0], self.bound[1][1], 20)
            z = np.linspace(self.bound[2][0], self.bound[2][1], 20)
            X, Y, Z = np.meshgrid(x, y, z)
                                 
            W = np.array([self.predict_state_given_param(np.array([x_val, y_val, z_val])).reshape(-1) for x_val, y_val, z_val in zip(((X - self.input_offset[0])/self.input_scale[0]).flatten(), 
                                                                                                                         ((Y - self.input_offset[1])/self.input_scale[1]).flatten(), 
                                                                                                                         ((Z - self.input_offset[2])/self.input_scale[2]).flatten())])
            # calculate the cost for each feature
            W = self.calc_stage_cost(W, target)  # calculate the stage cost
            # reshape W to match the grid shape
            W = W.reshape(X.shape[0], X.shape[1], X.shape[2], -1)
            
            
            for i in range(W.shape[3]): # i can only be 1
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=W[:, :, :, i].flatten(), cmap='viridis', marker='o', alpha=0.2)
                fig.colorbar(ax.collections[0], ax=ax, label=f'cost', shrink = 0.4, aspect=5)

                # Find max and min positions in W
                W_flat = W[:, :, :, i].flatten()
                max_idx = np.argmax(W_flat)
                min_idx = np.argmin(W_flat)
                max_pos = (X.flatten()[max_idx], Y.flatten()[max_idx], Z.flatten()[max_idx])
                min_pos = (X.flatten()[min_idx], Y.flatten()[min_idx], Z.flatten()[min_idx])

                # Mark max and min positions
                ax.scatter(*max_pos, color='red', marker='^', s=100, label='Max cost')
                ax.scatter(*min_pos, color='blue', marker='v', s=100, label='Min cost')
                ax.text(*max_pos, f"max: {W_flat[max_idx]:.2f}", color='red')
                ax.text(*min_pos, f"min: {W_flat[min_idx]:.2f}", color='blue')
                """ add trace to the figure """
                if trace is not None: 
                    x_trace =  np.array(trace["params"]).T 
                    # x_cost = np.array([1/2 * np.inner(w, (self.predict_state_given_param(p).reshape(-1) - target)**2) for p in x_trace])
                    # Add a trace for each parameter using matplotlib
                    ax.plot(x_trace[0,::skip], x_trace[1,::skip], x_trace[2,::skip], marker = ".", color='black', label='Trace') 
                    ax.plot(x_trace[0,-1], x_trace[1,-1], x_trace[2,-1], marker='*', color='green', label='Trace end', markersize=16) # mark the end parameter with text
                    # ax.text(x_trace[0,-1], x_trace[1,-1], x_trace[2,-1], f'({x_trace[0,-1]:.2f}, {x_trace[1,-1]:.2f}, {x_trace[2,-1]:.2f})', color='black')
                    # mark the start parameter with text
                    ax.text(
                        x_trace[0,-1], x_trace[1,-1], x_trace[2, -1],
                        f'({int(x_trace[0,-1])}, {int(x_trace[1,-1])}, {int(x_trace[2,-1])}), Cost: {trace["stage_cost"][-1]:.2f}',
                        color='black'
                    )
                    # mark the start parameter with text
                    ax.text(
                        x_trace[0,0], x_trace[1,0], x_trace[2, 0],
                        f'({int(x_trace[0,0])}, {int(x_trace[1,0])}, {int(x_trace[2,0])}), Cost: {trace["stage_cost"][0]:.2f}',
                        color='black'
                    )
                ax.set_xlabel(self.param_names[0] if self.param_names is not None else 'Parameter 1')
                ax.set_ylabel(self.param_names[1] if self.param_names is not None else 'Parameter 2')
                ax.set_zlabel(self.param_names[2] if self.param_names is not None else 'Parameter 3')
            
                ax.set_title(f'cost')

        else:
            raise ValueError("Model parameters should be 1, 2 or 3 dimensional for visualization.")
        
        plt.title('Simulator Model Visualization')
        plt.legend()
        plt.show()


    def vis_hist(self, his, skip = 1, target = None):
        # visualize all histories as subplots on the same figure

        fig, axes = plt.subplots(len(his)-1, 1, figsize=(6, 1.5 * (len(his)-1)), sharex=True)
        if len(his) == 1:
            axes = [axes]

        idx = 0
        for  (k, v) in his.items():
            if k != "next_state": 
                v = np.array(v)
                ax = axes[idx]
                if v.ndim == 1:
                    ax.plot(np.arange(0, v.shape[0],step = skip), v[::skip], marker='o', alpha=0.5)
                else:
                    for i in range(v.shape[1]):
                        if "param" in k:
                            lab = self.param_names[i]
                        elif "state" in k:
                            lab = self.state_names[i]
                        else:
                            lab = f'{k} {i+1}'
                        ax.plot(np.arange(0, v.shape[0],step = skip), v[::skip, i], marker='o', alpha=0.5, label=lab)
                        if 'state' in k and target is not None:
                            # mark the target state with horizontal line with the same color 
                            ax.axhline(y=target[i], color=ax.lines[-1].get_color(), linestyle='--', linewidth=0.5)
                    ax.legend(fontsize=8)
                idx += 1
            ax.set_title(f"History of {k} over iterations", fontsize=10)
            ax.set_ylabel(k, fontsize=9)
        axes[-1].set_xlabel("Iteration", fontsize=9)
        plt.tight_layout()
        plt.show()

    
    def summary(self):
        """ print the summary of the model """
        print("Simulator Model summary:")
        print("n_action:", self.n_action)
        print("n_state:", self.n_state)
        print("input_scale:", self.input_scale)
        print("output_scale:", self.output_scale)
        print("input_offset:", self.input_offset)
        print("output_offset:", self.output_offset)
        print("e_out:", self.e_out)
        print("e_in:", self.e_in)
        print("params:", self.params)  
        print("param_names:", self.param_names) 

class PiceSimulator(Simulator):
    def __init__(self, folder_path):
        """ Initialize the PICE simulator with the given folder path containing data files. """
        self.folder_path = folder_path
        
        # folder_path = "/Users/xinyi/Documents/Data/ossur/DC_05_18"
        params, state_names, states,  state_mean, state_std = load_states(folder_path)
        
            # specify by name:
        target_names = ["st_sw_phase", "min_knee_position_phase_st", "brake_time"]
        self.n_state = len(state_names)
        # filter states by names
        states = [s[:, [np.where(state_names == name)[0][0] for name in target_names]] for s in states]
        state_mean = np.array([np.mean(s, axis=0) for s in states])
        state_std = np.array([np.std(s, axis=0) for s in states])
        self.n_action = params.shape[1]
        self.state_names = np.array(target_names)
        print(f"state_names: {state_names}, mean: {state_mean.mean(axis=0)}")
        # for emulation scaling
        self.output_offset = np.mean(state_mean, axis=0)
        self.output_scale = np.std(state_mean, axis=0)
        self.input_offset = np.mean(params, axis=0)
        self.input_scale = np.std(params, axis=0)
        self.n_state = self.output_offset.shape[0]  # number of output features
        self.n_action = self.input_offset.shape[0]  # number of input features

        # normalize the feature (output), dimension is (n_samples, n_features)
        state_mean_normalized = (state_mean - np.mean(state_mean, axis=0)) / np.std(state_mean, axis=0) 
        # normalize the predictor (input), dimension is (n_samples, n_actions)
        params_normalized = (params - np.mean(params, axis=0)) / np.std(params, axis=0)
        _, model = feature_regression_analysis(state_mean_normalized, 
                                                state_std, 
                                                self.state_names, 
                                                params_normalized, 
                                                param_names=["Init STF angle", "SWF target", "SW init"],
                                                vis = False)
        # model.summary()
        # create the simulator from the model
        super().__init__(model, input_scale=self.input_scale,
                                input_offset=self.input_offset,
                                output_scale=self.output_scale,
                                output_offset=self.output_offset)
    


def case_LS_fixed_action():
    """ Example case for the simulator with a simple linear model with 3 inputs and 2 outputs."""
    # Example usage
    # Load a model, e.g., from a file or create one
    # y = np.array([1, 2, 3, 4, 5]) # dof model is 1, output dimension is 1
    y = np.array([[1, 2, 3, 4, 5],[-1, -2, -3, -4, -5]]).T # (n_sample, n_out) output dimension is 2
    # X = np.array([[1, 1, -1], [1, 2, -2], [3, 3, -3], [-3, 4, -4], [5, 5, -5]]) # # (n_sample, n_in) input dimension is 3
    X = np.array([1,2,3,4,5])
    action = np.array([0.1])#, -0.1, -0.2]) # abs scale, same dimension as n_in

    # normalize input and output
    output_offset = np.mean(y, axis=0)
    input_offset = np.mean(X, axis=0)
    output_scale = np.std(y, axis=0)
    input_scale = np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y, axis=0)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    weights = np.array([1, 1, 1, 1, 1])  # Example weights
    model = sm.WLS(y, X, weights=weights).fit()
    

    # create the simulator from the model
    simulator = Simulator(model, input_scale=input_scale,
                            input_offset=input_offset,
                            output_scale=output_scale,
                            output_offset=output_offset)
    
    # print("m.n_action, m.n_state",m.n_action, m.n_state)
    # Reset the simulator
    simulator.reset()
    
    history = {"params": [], "next_state": []}  # history of next state and parameters
    # Predict with an action
    for _ in range(10):
        
        # print("Action:", action)
        next_state = simulator.step(action)
        # Predict the next state
        next_state = simulator.predict()
        history["params"].append(simulator.params.copy())
        history["next_state"].append(next_state.copy())
        print("param history:", history["params"][-1], "Next state:", next_state, "Action:", action)
    
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
    simulator = PiceSimulator(folder_path=folder_path)
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
           "param_at_bound": []} # indicates if the parameters are at the bound
    # External loop for policy update iterations
    for k in range(20):  # for each policy update iteration
        
        print(f"Policy update iteration {k+1}")
        for i in range(20):  # for each trial 

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
            state = next_state

        # Visualize after each policy update
        target = simulator.output_offset #np.array([0., 0., 0.])  # target state for visualization
        
        
        # update policy 
        ssa = np.concatenate([his["current_state"].copy() - target, his["next_state"].copy() - target, his["action"].copy()], axis=1)
        ssar = add_quadratic_cost_stack(ssa, simulator.n_action)
        samples = load_from_data(ssar, simulator.n_state, simulator.n_action)
        solver = LSTDQSolver()#PICESolver()
        policy, all_policies = lspi_loop_offline(solver, samples, initial_policy=policy,
                                                    discount=0.9,
                                                    epsilon=0.001, 
                                                    max_iterations=1,
                                                    verbose=False)
        # print(f"Updated policy weights: {np.array2string(convertW2S(policy.weights), formatter={'float_kind':lambda x: '%.4f' % x})}")
        
        
        
        # exit condition: minimum stage cost
        if k > 0 and np.mean(his["stage_cost"][-5:]) < 0.2:  # check if the policy has converged
            print(f"At i = {k}, global minimum reached, stopping training.")
            if vis: 
                simulator.vis_cost(target, trace=his)
                simulator.vis_hist(his, target=target) 
            return True, his
        # reset conditions:
        elif k > 0 and np.mean(his["param_at_bound"][-5:]) > 0.8:
            print(f"At i = {k}, Parameters at bound, resetting simulator.")
            simulator.reset_to_hist_best_or_random(init_params,  his, his_window = 10)
            state = simulator.predict()
            if vis:
                simulator.vis_cost(target, trace=his)
                simulator.vis_hist(his, target=target)     
        elif k>0 and np.linalg.norm(policy.weights - all_policies[-2].weights) < 0.1:
            print(f"At i = {k},convergence to a local minimum, not global, at p = 0.5, simulator reset.")
            if np.random.random()< 0.5:
                print("resetting")
                simulator.reset_to_hist_best_or_random(init_params,  his, p_init = 0.1, p_random = 0.2, p_pass = 0.5)
                
                # state = simulator.predict()
                # print(f"Resetting to historic best params {simulator.params} with stage cost {np.mean(his['stage_cost'][-5:]):.4f}")
            # simulator.reset_to_hist_best_or_random(init_params,  his)
            
            
    
    # visualize the final policy
    if vis:
        # simulator.vis_model(trace=his, n_grid=20)
        simulator.vis_cost(target, trace=his, n_grid=20)
        simulator.vis_hist(his, skip = 1, target=target)  # visualize the history of the policy
        
    return False, his 

def case_DDPG(vis=True):
    from ddpg_agent import DDPGAgent
    import torch
    folder_path = "/Users/xinyi/Documents/Data/ossur/DC_05_18"
    simulator = PiceSimulator(folder_path=folder_path)
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
        

def eval_case(func, n_eval,  vis = False, **kwargs):
    from tqdm import tqdm
    """ Evaluate a case function with randomized init params """
    bounds = np.array([[40., 75.],[0., 100.],[0., 100.]])  # bounds for the parameters, normalized
    results = {"success": [], 
               "final stage cost": [], 
               "cost_reduction_ratio": [],
               "init_params": []}  # store success results
    for i in tqdm(range(n_eval), desc="Evaluations"):
        init_params = np.random.uniform(bounds[:, 0], bounds[:, 1])
        print(f"Evaluation {i+1}/{n_eval}, init_params: {init_params}")
        success, his = func(vis=vis, init_params=init_params, **kwargs)
        results["init_params"].append(init_params)
        results["success"].append(success)
        results["final stage cost"].append(np.mean(his["stage_cost"][-5:]))
        results["cost_reduction_ratio"].append(np.mean(his["stage_cost"][-5:]) / np.mean(his["stage_cost"][:5]))  # cost reduction ratio

    
    success_rate = np.mean(results["success"])
    print(f"Success rate: {success_rate:.2f} ({np.sum(results['success'])}/{n_eval})")
    print(f"Average final stage cost: {np.mean(results['final stage cost']):.4f}")
    print(f"Average cost reduction ratio: {np.mean(results['cost_reduction_ratio']):.4f}")

    for k, v in results.items():
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
    
    # find if the init is directly associated with success scatter 3d plot
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(results["init_params"][:, 0], results["init_params"][:, 1], results["init_params"][:, 2], 
                c=results["success"], cmap='viridis', marker='o', alpha=0.5)
    # Add color bar
    cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.1)
    ax.set_xlabel(Simulator.param_names[0] if hasattr(Simulator, 'param_names') else 'Param 1')
    ax.set_ylabel(Simulator.param_names[1] if hasattr(Simulator, 'param_names') else 'Param 2')
    ax.set_zlabel(Simulator.param_names[2] if hasattr(Simulator, 'param_names') else 'Param 3')

    plt.title("Success Scatter Plot")
    plt.show()


    

 

if __name__ == "__main__":

    # case_LS_fixed_action()

    # case_PICE(vis = False)
    eval_case(case_PICE, n_eval=20, vis = False)  # evaluate the PICE case with 10 evaluations

    # case_DDPG()

   

        
       

    



    
    
