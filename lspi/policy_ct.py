# -*- coding: utf-8 -*-
"""LSPI Policy class for continuous state/action spaces."""
import copy
import random
import numpy as np
from scipy.optimize import minimize
import os
import sys
# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from policy import Policy
from basis_functions import QuadraticBasisFunction




class QuadraticPolicy(Policy):
    """Implements LSPI policy with quadratic programming for continuous control."""
    
    def __init__(self, n_state, n_action, discount=1.0, explore=0.0, weights=None, 
                  folder_path = None):
        self.n_action = n_action
        self.n_state = n_state
        self.n_basis = int((n_action + n_state)*(n_state + n_action + 1)/2)
        print(f"Number of basis functions: {self.n_basis}")
        self.basis = QuadraticBasisFunction(n_state, n_action)
 

        if folder_path is not None:
            # load scaling and offset for policy 
            self.offset_a = np.load(os.path.join(folder_path, "offset_a.npy"))
            self.scale_a = np.load(os.path.join(folder_path,"scale_a.npy"))
            self.offset_s = np.load(os.path.join(folder_path,"offset_s.npy"))
            self.scale_s = np.load(os.path.join(folder_path,"scale_s.npy"))
        else:
            self.scale_s = 1.
            self.scale_a = 1.
            self.offset_s = 0.
            self.offset_a = 0.
        
        super().__init__(self.basis, discount, explore, weights)
        

        
    def cp(self):
        """Return a copy of this class with a deep copy of the weights."""
        return QuadraticPolicy(self.n_state, 
                        self.n_action,
                        self.discount,
                        self.explore,
                        np.copy(self.weights),
                        scale_s=self.scale_s,
                        scale_a=self.scale_a,
                        offset_s=self.offset_s,
                        offset_a=self.offset_a
                       )
    
    def calc_q_value(self, state, action):
        # scale and offset state and action
        state = (state - self.offset_s) / self.scale_s
        action = (action - self.offset_a) / self.scale_a
        """Calculate Q-value for a given state-action pair."""
        return self._calc_q_value(state, action)
    
    def _calc_q_value(self, state, action):
        # if action.shape[0] < 0 or action >= self.n_action:
        #     raise IndexError('action must be in range [0, num_actions)')
        self.Huu, self.Hf = self.extract_qp_parameters(state)
        q = - 0.5 * action[None,:] @ self.Huu @ action[:,None] - self.Hf @ action[:,None]

        return q # self.weights.dot(self.basis.evaluate(state, action))

    def update_state_dependent_constraints(self, first_stiff_timing, second_stiff_timing):
        self.first_stiff_timing = first_stiff_timing
        self.second_stiff_timing = second_stiff_timing

    def select_action(self, state):
        """Select action with ε-greedy exploration."""
        if random.random() < self.explore:
            return np.random.uniform(low=-0.12, high=0.12, size=self.n_action)
        return self.best_action(state)
    
    def best_action(self, state):
        """apply scaling and offsetting of action and state"""
        state = (state - self.offset_s) / self.scale_s
        action = self._best_action(state)
        action = action * self.scale_a + self.offset_a
        return action

    def _best_action(self, state):
        self.Huu, self.Hf = self.extract_qp_parameters(state) 


        # A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Action bounds
        
        # Ax = np.array([
        #     -0.01 + self.first_stiff_timing,
        #     0.5 - self.first_stiff_timing,
        #     -0.5 + self.second_stiff_timing,
        #     0.98 - self.second_stiff_timing
        # ]) # Action bounds
        
        # Solve constrained optimization
        res = minimize(
            fun=lambda a: 0.5 * a[None,:] @ self.Huu @ a[:,None] + self.Hf @ a[:,None], 
            x0=np.zeros((self.n_action,)),
            bounds=[(-1, 1)], # for a N(0,1) distribution, it is reasonable to set the bounds to (-3, 3), because 99.7% of the values will fall within this range
            # constraints={'type': 'ineq', 'fun': lambda a: A @ a - Ax}
        )
        action = res.x
        
        # Add exploration noise
        # exploration = False
        # if exploration:
        #     noise_scale = max(0, 1 - self.rl_number/3)
        #     action += noise_scale * np.random.uniform(-0.15, 0.15, size=action.shape)
        
        # Apply post-optimization constraints
        # action = self._apply_action_constraints(action)
        return action

    def extract_qp_parameters(self, state):
        """Solve quadratic program for optimal action with constraints."""
        nS = self.n_state  # State dimensions
        nSA = self.n_state + self.n_action  # State+action dimensions
        
        HW = convertW2S(self.weights)
        # Extract weight submatrices
        Hux = HW[nS:nSA, :nS] 
        Huu = HW[nS:nSA, nS:nSA]
        Hxu = HW[:nS, nS:nSA]
        
        # Quadratic programming setup
        Hf = state.T @ Hxu
        return Huu,Hf

    def _apply_action_constraints(self, action):
        """Enforce physical constraints on actions."""
        y1, y2 = self.first_stiff_timing, self.second_stiff_timing
        
        if action[0] + y1 < 0.01:
            action[0] = 2 * abs(y1 - 0.01)
            
        if (y2 + action[1] - y1 - action[0]) < 0:
            action[1] = abs(y2 - y1 - action[0]) * 2
            
        if action[1] + y2 > 0.98:
            action[1] = -2 * abs(0.98 - y2)
            
        return np.clip(action, -1, 1)

    def evaluate_basis(self, state, action):
        """Evaluate basis function for given state-action pair."""
        return self.basis(state, action)



def convertW2S(w):
    """Convert weight vector to a symmetric matrix."""
    # Size of the symmetric matrix
    n = int((np.sqrt(1 + 8 * len(w)) - 1) / 2) 
    idx = 0
    Phat = np.zeros((n, n))
    
    # Fill the upper triangular part of the matrix
    for r in range(n):
        for c in range(r, n):
            Phat[r, c] = w[idx]
            idx += 1
    
    # Symmetrize the matrix
    S = 0.5 * (Phat + Phat.T)
    return S