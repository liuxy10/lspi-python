import numpy as np
# import tkinter as tk
# from tkinter import ttk
import matplotlib.pyplot as plt

class InvertedPendulumGUI:
    def __init__(self, root, length=1.0, mass=1.0, gravity=9.81, dt=0.01, force_max=10.0):
        self.length = length
        self.mass = mass
        self.gravity = gravity
        self.dt = dt
        self.force_max = force_max
        self.randomize_state()
        

    def randomize_state(self):
        """Randomize the initial state of the pendulum"""
        self.state = np.random.uniform(-np.pi/3, np.pi/3, size=(2,))
    def dynamics(self, s, a):
        angle, ang_vel = s
        force = np.clip(a, -self.force_max, self.force_max)
        ang_acc = (self.gravity/self.length)*np.sin(angle) + force/(self.mass*self.length**2)
        return ang_acc  # Remove [1][0] indexing


    def reset(self):
        self.running = False
        self.state = np.array([0.0, 0.0])  # Reset to upright position
        # self.draw_pendulum()



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
    
    pendulum = InvertedPendulumGUI(None, length, mass, gravity, dt, force_max)
    

    for _ in range(int(n_samples)):
        pendulum.randomize_state()
        for _ in range(n_step_per_episodes):
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
            rewards.append(-pendulum.state[0]**2 - pendulum.state[1]**2 - action**2)  # Placeholder for reward
            next_states.append(pendulum.state.copy())

    if vis:
        #  visualize the states
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(0, dt * n_samples * n_step_per_episodes, dt), np.array(states)[:, 0], label='Angle')
        ax.plot(np.arange(0, dt * n_samples * n_step_per_episodes, dt), np.array(rewards), label='Reward')
        ax.plot(np.arange(0, dt * n_samples * n_step_per_episodes, dt), np.array(states)[:, 1], label='Angular Velocity')
        ax.hlines(0, 0, dt * n_samples * n_step_per_episodes, colors='r', linestyles='dashed', label='Zero Line')
        ax.plot(np.arange(0, dt * n_samples * n_step_per_episodes, dt), np.array(actions), label='Action')
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
    s,s1, a,r = generate_ssar(n_samples=10, n_step_per_episodes=50,dt=0.02, vis=True)

    
    ssar = np.concatenate((s, s1, a.reshape(-1, 1), r.reshape(-1, 1)), axis=1)
    print("States:", s.shape)
    print("Actions:", a.shape)
    print("Rewards:", r.shape)



    from utils import *
    from lspi.solvers import LSTDQSolver, PICESolver
    from lspi.train_offline import lspi_loop_offline
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

    
