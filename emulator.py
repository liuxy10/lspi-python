
import numpy as np
from lspi.policy_ct import QuadraticPolicy
from data_processing_offline import load_states
from utils import *
import statsmodels.api as sm

class Emulator:
    """ use sm model as a emulator to generate samples
    model: sm.WLSModel or sm.OLSModel
    e_out: error in output
    e_in: error in input
    n_action: number of actions
    n_state: number of states
    params: parameters of the emulator, shape (n_action, )
    """

    def __init__(self, model, e_out = 0.01, e_in = 0.01, 
                 input_scale = 1., input_offset = 0, 
                 output_scale = 1., output_offset = 0):
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
        self.n_state = input_scale.shape[0] if input_scale is not int else model.model.endog.shape[1] # number of output features in the model
        self.params = np.zeros(self.n_action) 
        self.summary()



    def reset(self):
        """ reset the emulator """
        self.params = np.zeros(self.n_action) # change

    def reset_params(self, params):
        assert params.shape == (self.n_action,), "Params shape must match the number of actions"
        """ set the parameters of the emulator """
        self.params = (params - self.input_offset) / self.input_scale  # normalize the parameters

    def step(self, action):
        assert action.shape == (self.n_action,), "Action shape must match the number of actions"
        """ predict the next state and reward given the action """
        self.params += action / self.input_scale

    def predict(self):
        """ predict the next state and reward given the current parameters (scaled and offset) """
        # Add constant term for prediction if needed
        params_with_noise = self.params + self.e_in * np.random.randn(self.n_action)
        if "const" in self.model.model.exog_names:
            params_with_noise = np.concatenate(([1.0], params_with_noise))
        pred = self.model.predict(params_with_noise)
        return self.e_out * np.random.randn(self.n_state) + pred * self.output_scale + self.output_offset
    
    def vis_model(self, trace = None):
        """ visualize the model in normalized space
            plot each feature againt the parameters
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
        
        # sweep each parameter
        if self.n_action == 1:
            params_sweep = np.linspace(-2,2, 40) # sweep by 40
            # predict features
            y = np.array([self.model.predict(p).reshape(-1) for p in params_sweep])
            for i in range(y.shape[1]): # plot each feature
                plt.plot(params_sweep, y[:,i], label=f'Feature {i+1}')
                plt.scatter(params_sweep, y[:,i], s=10, alpha=0.5)
            plt.xlabel('X')
            plt.ylabel('y')
            
        elif self.n_action == 2:
            # build meshgrid for 2D parameters
            x = np.linspace(-2, 2, 40)
            y = np.linspace(-2, 2, 40)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[self.model.predict(np.array([x_val, y_val])).reshape(-1) for x_val in x] for y_val in y])
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
            W = np.array([self.model.predict(np.array([x_val, y_val, z_val])).reshape(-1) for x_val, y_val, z_val in zip(X.flatten(), Y.flatten(), Z.flatten())])
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
                ax.set_xlabel('Parameter 1')
                ax.set_ylabel('Parameter 2')
                ax.set_zlabel('Parameter 3')
                ax.set_title(f'Feature {i+1} vs Parameters')

        else:
            raise ValueError("Model parameters should be 1, 2 or 3 dimensional for visualization.")
        
        plt.title('Emulator Model Visualization')
        plt.legend()
        plt.show()
        return fig,ax


    def summary(self):
        """ print the summary of the model """
        print("Model summary:")
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

class PiceEmulator(Emulator):
    def __init__(self, folder_path):
        """ Initialize the PICE emulator with the given folder path containing data files. """
        self.folder_path = folder_path
        
        # folder_path = "/Users/xinyi/Documents/Data/ossur/DC_05_18"
        params, state_names, states,  state_mean, state_std = load_states(folder_path)
        
            # specify by name:
        target_names = ["st_sw_phase", "toe_off_time", "brake_time"]
        self.n_state = len(state_names)
        # filter states by names
        states = [s[:, [np.where(state_names == name)[0][0] for name in target_names]] for s in states]
        state_mean = np.array([np.mean(s, axis=0) for s in states])
        state_std = np.array([np.std(s, axis=0) for s in states])
        self.n_action = params.shape[1]
        state_names = np.array(target_names)
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
                                                    state_names, 
                                                    params_normalized, 
                                                    param_names=["Init STF angle", "SWF target", "SW init"],
                                                    vis = False)
        model.summary()
        # create the emulator from the model
        super().__init__(model, input_scale=self.input_scale,
                                input_offset=self.input_offset,
                                output_scale=self.output_scale,
                                output_offset=self.output_offset)
    
    
    def rand_init_state(self):
        return np.random.randn(self.n_state) * self.output_scale + self.output_offset

    def vis_cost(self, trace=None):
        """ visualize the cost function in normalized space
            plot each feature against the parameters
        """
        pass


def case_IO_32():
    """ Example case for the emulator with a simple linear model with 3 inputs and 2 outputs."""
    # Example usage
    # Load a model, e.g., from a file or create one
    # y = np.array([1, 2, 3, 4, 5]) # dof model is 1, output dimension is 1
    y = np.array([[1, 2, 3, 4, 5],[-1, -2, -3, -4, -5]]).T # (n_sample, n_out) output dimension is 2
    X = np.array([[1, 1, -1], [1, 2, -2], [3, 3, -3], [-3, 4, -4], [5, 5, -5]]) # # (n_sample, n_in) input dimension is 3
    # X = np.array([1, 2,3,4,5])

    # normalize input and output
    output_offset = np.mean(y, axis=0)
    input_offset = np.mean(X, axis=0)
    output_scale = np.std(y, axis=0)
    input_scale = np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y, axis=0)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    weights = np.array([1, 1, 1, 1, 1])  # Example weights
    model = sm.WLS(y, X, weights=weights).fit()
    

    # create the emulator from the model
    emulator = Emulator(model, input_scale=input_scale,
                            input_offset=input_offset,
                            output_scale=output_scale,
                            output_offset=output_offset)
    
    # print("m.n_action, m.n_state",m.n_action, m.n_state)
    # exit()
    # Reset the emulator
    emulator.reset()
    
    history = []
    # Predict with an action
    for _ in range(10):
        action = np.array([0.1, -0.1, -0.2]) # abs scale
        # print("Action:", action)
        next_state = emulator.step(action)
        # Predict the next state
        next_state = emulator.predict()
        history.append(emulator.params.copy())
        print("param history:", history[-1], "Next state:", next_state)

    fig, ax = emulator.vis_model(history)


def case_PICE():
    folder_path = "/Users/xinyi/Documents/Data/ossur/DC_05_18"
    # emulator 
    emulator = PiceEmulator(folder_path=folder_path)
    # load policy
    w = np.load("policy_weights.npy")
    policy = QuadraticPolicy(n_state=3, n_action=3, weights=w,
                              explore = 0.01, 
                              discount = 0.9,
                              folder_path = folder_path) # folder is used to load calculated scale and offset
    
    his = {"next_state": [], "params": []}  # history of next state and parameters
    emulator.reset_params(np.array([60, 60, 60]))  # reset the emulator parameters
    state = emulator.rand_init_state()  # get a random initial state

    for i in range(100):

        action = policy.best_action(state) 
        print("Action:", action)
        emulator.step(action)
        next_state = emulator.predict()
        his["params"].append(emulator.params.copy())
        his["next_state"].append(next_state.copy())
        print("Next state:", next_state, "Params:", emulator.params, "cost", policy.calc_q_value(state, action))
        state = next_state
    
    vis = True
    if vis: 
        # vis the his
        for k,v in his.items():
            v = np.array(v)
            fig = plt.figure(figsize=(10, 8))
            plt.plot(v, marker='o', alpha=0.5)
            plt.title(f"History of {k} over iterations")
            plt.xlabel("Iteration")
            plt.ylabel(k)
            plt.show()





if __name__ == "__main__":

    # case_IO_32()
    case_PICE()

   

        
       

    



    
    
