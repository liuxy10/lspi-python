   

import gym
from gym import spaces
import numpy as np

class SimulatorEnv(gym.Env):
    def __init__(self, simulator, target_state, max_steps=100, cost_threshold=None):
        super(SimulatorEnv, self).__init__()
        
        self.simulator = simulator
        self.target = target_state
        self.max_steps = max_steps
        self.cost_threshold = self.simulator.param_region if cost_threshold is None else cost_threshold
        self.current_step = 0
        self.current_cost = np.inf
        self.last_state = None
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-5.0, 
            high=5.0,
            shape=(self.simulator.n_action,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.simulator.n_state,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.current_cost = np.inf
            

    def reset(self):
        """Reset the environment to initial state"""
        params = np.int32( np.random.uniform(
            low=self.simulator.bound[:, 0],
            high=self.simulator.bound[:, 1]
        ))
        self.simulator.reset_params(params)
        self.current_step = 0
        self.last_state = self.simulator.predict()
        self.history = {
        "current_state": [],
        "next_state": [],
        "params": [],
        "action": [],
        "stage_cost": [],
        "param_at_bound": [],
        "start": []
        }

        return self.last_state

    def step(self, action):
        """Execute one time step in the environment"""
        self.current_step += 1
        
        # Apply action to simulator
        self.simulator.step(action)
        
        # Get new state
        next_state = self.simulator.predict()
        
        # Calculate reward (negative stage cost)
        self.current_cost = self.simulator.calc_stage_cost(next_state, self.target)
        reward = -self.current_cost
        
        # Check termination conditions
        done = (self.current_cost < self.cost_threshold) or (self.current_step >= self.max_steps)
        
        # Additional info (optional)
        info = {
            'params': self.simulator.get_params(),
            'stage_cost': self.current_cost,
            'param_at_bound': self.simulator.at_bound(),
            'current_state': self.last_state,
            'action': action.copy()
        }
        self.last_state = next_state
        # If done, reset the simulator
        if done:
            self.simulator.reset()
            self.current_step = 0
            self.last_state = self.simulator.predict()
            self.current_cost = np.inf
        return next_state, reward, done, info

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            self.simulator.vis_cost(self.target,self.history)
            return self.simulator.vis_hist(self.history, target=self.target)
        elif mode == 'rgb_array':
            return self.simulator.vis_hist(self.history)
            



    def close(self):
        return super().close()
