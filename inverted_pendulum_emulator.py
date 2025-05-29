import numpy as np
# import tkinter as tk
# from tkinter import ttk
import matplotlib.pyplot as plt

import os
import sys
# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Inverted_Pendulum import InvertedPendulum


def generate_ssar(n_samples=100, n_step_per_episodes = 100, dt=0.01, policy = None, vis = False,
                  length = 1.0,
                    mass = 1.0,
                    gravity = 9.81,
                    force_max = 5.):
    """Generate state, action, reward, next_state tuples"""
    

    states = []
    actions = []
    rewards = []
    next_states = []
    reset_flag = []
    
    pendulum = InvertedPendulum(None, length, mass, gravity, dt, force_max)
    

    for _ in range(int(n_samples)):
        pendulum.randomize_state(max_angle=0.5)
        for i in range(n_step_per_episodes):
            if policy is None:
                action = np.inner(np.array([-.5, -.3]), np.sign(pendulum.state)) * mass * gravity# Random action
            else:
                action = policy.select_action(pendulum.state)
            pendulum.state[1] += pendulum.dynamics(pendulum.state, action) * dt
            pendulum.state[0] += pendulum.state[1] * dt + np.random.normal(0, 0.01) * dt
            if pendulum.state[0] > np.pi:
                pendulum.state[0] -= 2 * np.pi
            elif pendulum.state[0] < -np.pi:
                pendulum.state[0] += 2 * np.pi
            pendulum.state[1] = np.clip(pendulum.state[1], -10, 10)
            pendulum.state[0] = np.clip(pendulum.state[0], -np.pi, np.pi)
            # Store the state, action, reward, and next state
            
            states.append(pendulum.state.copy())
            actions.append(action)
            rewards.append(-pendulum.state[0]**2 - 0.1 * pendulum.state[1]**2) # action**2)  # Placeholder for reward
            next_states.append(pendulum.state.copy())

            # done
            if pendulum.state[0] > np.pi/2 or pendulum.state[0] < -np.pi/2 or i == n_step_per_episodes - 1:
                reset_flag.append(1)
                # print("Resetting the pendulum")
                break
            else:
                reset_flag.append(0)
            # Update the GU
            

    if vis:
        #  visualize the states
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(0, dt * len(states), dt), np.array(states)[:, 0], label='Angle')
        ax.plot(np.arange(0, dt * len(states), dt), np.array(rewards), label='Reward')
        ax.plot(np.arange(0, dt * len(states), dt), np.array(states)[:, 1], label='Angular Velocity')
        ax.hlines(0, 0, dt * len(states), colors='r', linestyles='dashed', label='Zero Line')
        # ax.plot(np.arange(0, dt * len(states), dt), np.array(actions)/10, label='Action/10')
        for i in range(len(reset_flag)):
            if reset_flag[i] == 1:
                ax.axvline(x=i*dt, color='g', linestyle='--')

        ax.set_xlabel('Time')
        ax.set_xlabel('Angle')
        ax.set_ylabel('Angular Velocity')
        ax.set_title('State Space')
        ax.legend()
        plt.grid()

        plt.show()
        
    

    return np.array(states),  np.array(next_states), np.array(actions), np.array(rewards),




if __name__ == "__main__":
    # run_gui()
    s,s1, a,r = generate_ssar(n_samples=3, n_step_per_episodes=200,dt=0.02, vis=True)

    
    ssar = np.concatenate((s, s1, a.reshape(-1, 1), r.reshape(-1, 1)), axis=1)
    print("States:", s.shape)
    print("Actions:", a.shape)
    print("Rewards:", r.shape)



    from utils import *
    from lspi.solvers import LSTDQSolver, PICESolver
    from lspi.lspi_train_offline import lspi_loop_offline
    from lspi.policy_ct import convertW2S, QuadraticPolicy


    samples = load_from_data(ssar, n_state=s.shape[1], n_action=1) 
    # solver = LSTDQSolver()
    solver = PICESolver()
    init_policy = QuadraticPolicy(n_state=s.shape[1], n_action=1, discount=0.8, explore=0.01, weights= np.array([0.5, 0.3, 0., 0., 0., 0.]).reshape(-1,)*9.81)
    # Convert weights to state-action space
    policy, all_policies = lspi_loop_offline(solver, samples, discount=0.8, epsilon=0.01, max_iterations=1, initial_policy=init_policy)
    print("Policy weights:", convertW2S(policy.weights))
    
    for i in range(10):
        generate_ssar(n_samples=1, n_step_per_episodes=300, dt=0.05, policy=policy, vis=True)
        

    # testing the policy

    
