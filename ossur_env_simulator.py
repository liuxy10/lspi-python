
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
                 bound = None,  # bounds for the parameters, normalized for our case
        verbose = False
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
        self.bound = bound if bound is not None else np.array([self.input_offset + 3 * self.input_scale, self.input_offset - 3 * self.input_scale]) # bounds for the parameters, normalized, if not provided, use 3 standard deviations from the mean
            
        self.verbose = verbose
        if self.verbose:
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
        if isinstance(self.input_offset, np.ndarray):
            bound = (self.bound - self.input_offset[:, None]) / self.input_scale[:, None]
            for i in range(self.n_action):
                if self.params[i] < bound[i, 0]:
                    self.params[i] = bound[i, 0]
                elif self.params[i] > bound[i, 1]:
                    self.params[i] = bound[i, 1]
        else: # n_action == 1
            bound = (self.bound - self.input_offset) / self.input_scale
            if self.params < bound[0]:
                self.params = bound[0]
            elif self.params > bound[1]:
                self.params = bound[1]
       
    
    def at_bound(self, margin = 0.01):
        """ check if the parameters are at the bound """
        i = 0
        for i in range(self.n_action):
            if self.params[i] < (self.bound[i, 0] - self.input_offset[i]) / self.input_scale[i] + margin or \
               self.params[i] > (self.bound[i, 1] - self.input_offset[i]) / self.input_scale[i] - margin:
                i+= 1
        return i
    
    def _calc_stage_cost(self, y, target):
        return 1/2 * np.inner(self.weights, (y - target)**2) ## TODO: add action weights R
    
    
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

    def vis_cost(self, target, trace=None, n_grid = 20, skip = 1):
        """ visualize the quadratic cost function in normalized space
            plot each feature against the parameters
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
        assert target.shape == (self.n_state,), "Target shape must match the number of states"
        
        w = self.weights if self.weights is not None else np.ones(self.n_state)  # use weights for cost function, default to 1 for each feature
        # sweep each parameter
        if self.n_action == 1:
            params_sweep = np.linspace(-2,2, n_grid) # sweep by n_grid
            # predict features
            y = np.array([self.predict_state_given_param(p).reshape(-1) for p in params_sweep]) # predict the features
            cost = 1/2 * np.inner(w, (y - target)**2)  # calculate the stage cost but without action cost
            
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
            W = np.log10(np.clip(self.calc_stage_cost(W, target), 1e-8, 1000))  # calculate the stage cost, need to be consistent with quad cost function
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
            
                ax.set_title(f'cost in log scale color vs Parameters trace')

        else:
            raise ValueError("Model parameters should be 1, 2 or 3 dimensional for visualization.")
        
        plt.title('Simulator Model Visualization')
        plt.legend()
        plt.show()

    def vis_hist(self, his, skip=1, target=None, keys=None):
        # If keys not specified, plot all except 'next_state' and 'start'
        if keys is None:
            keys = [k for k in his.keys() if k not in ("next_state", "start")]

        fig, axes = plt.subplots(len(keys), 1, figsize=(6, 1.5 * len(keys)), sharex=True)
        if len(keys) == 1:
            axes = [axes]

        for idx, k in enumerate(keys):
            v = np.array(his[k])
            ax = axes[idx]
            if "start" in his.keys():  # print vert line to separate different rl trials
                start_indices = np.where(np.array(his["start"]) == 1)[0]
                for start_idx in start_indices:
                    ax.axvline(x=start_idx, color='red', linestyle='--', linewidth=0.5)
            if v.ndim == 1:
                ax.plot(np.arange(0, v.shape[0], step=skip), v[::skip], marker='o', alpha=0.5)
            else:
                for i in range(v.shape[1]):
                    if "param" in k:
                        lab = self.param_names[i]
                    elif "state" in k and hasattr(self, "state_names"):
                        lab = self.state_names[i]
                    else:
                        lab = f'{k} {i+1}'

                    if "state" in k and target is not None:
                        ax.plot(np.arange(0, v.shape[0], step=skip), v[::skip, i] - target[i], marker='o', alpha=0.5, label=lab)
                        ax.axhline(y=self.param_region, linestyle='--', linewidth=0.5)
                        ax.axhline(y=-self.param_region, linestyle='--', linewidth=0.5)
                    else:
                        ax.plot(np.arange(0, v.shape[0], step=skip), v[::skip, i], marker='o', alpha=0.5, label=lab)
                ax.legend(fontsize=8)
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






class LRSimulator(Simulator): # linear regression simulator
    """ A linear regression simulator that uses a weighted least squares model to predict the next state and reward given the action.
        The model is trained on the data from the folder_path.
        The model is used to predict the next state and reward given the action.
        The model is used to calculate the stage cost.
    """
    def __init__(self, folder_path):
        """ Initialize the PICE simulator with the given folder path containing data files. """
        
        # Specify target state names
        target_names = np.array(["st_sw_phase", "min_knee_position_phase_sw", "brake_time"])
        bound = np.array([[45, 75], [0, 100], [0, 100]])
        self.n_state = len(target_names)
        self.state_names = np.array(target_names)
        if folder_path is not None: # load data from the folder
            assert os.path.exists(folder_path), f"Folder {folder_path} does not exist."
            self.folder_path = folder_path
            # Load and preprocess data
            params, state_names, states, state_mean, state_std = load_states(folder_path)
            
        else: # simulate with random correlated data, 
            # Generate random data for testing
            np.random.seed(42)
            n_samples = 1000
            params = np.random.uniform(bound[:, 0], bound[:, 1], (n_samples, 3))  # 3 parameters, uniformly random in bound
            state_names = target_names
            # Create states as noisy linear combinations of params
            states = []
            for i in range(3):
                # Each state is a linear combination of params plus noise
                weights = np.random.uniform(0.5, 1.5, (3, self.n_state))
                bias = np.random.uniform(-0.2, 0.2, self.n_state)
                noise = np.random.normal(0, 0.05, (n_samples, self.n_state))
                state = params @ weights + bias + noise
                states.append(state)
            state_mean = np.mean(states, axis=1)
            state_std = np.std(states, axis=1)
        

        # Filter and normalize states
        state_indices = [np.where(state_names == name)[0][0] for name in target_names]
        states = [s[:, state_indices] for s in states]
        state_mean = np.array([np.mean(s, axis=0) for s in states])
        state_std = np.array([np.std(s, axis=0) for s in states])

        # Set input/output scaling and offsets
        self.input_offset = np.mean(params, axis=0)
        self.input_scale = np.std(params, axis=0)
        self.output_offset = np.mean(state_mean, axis=0)
        self.output_scale = np.std(state_mean, axis=0)
        self.n_action = self.input_offset.shape[0]
        self.param_region = 0.02

        # Normalize features and predictors
        state_mean_normalized = (state_mean - self.output_offset) / self.output_scale
        params_normalized = (params - self.input_offset) / self.input_scale

        # Fit regression model
        _, model = feature_regression_analysis(
            state_mean_normalized,
            state_std,
            self.state_names,
            params_normalized,
            param_names=["Init STF angle", "SWF target", "SW init"],
            vis=False
        )
        # model.summary()
        # create the simulator from the model
        super().__init__(model, input_scale=self.input_scale,
                                input_offset=self.input_offset,
                                output_scale=self.output_scale,
                                output_offset=self.output_offset, bound = bound,)
    

    