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
        self.state = np.random.random(2)  # [angle, angular velocity]
        
        # GUI setup
        # self.root = root
        # self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        # self.force_slider = ttk.Scale(root, from_=-force_max, to=force_max)
        # self._setup_controls()
        
    # def _setup_controls(self):
    #     """Create interactive GUI elements"""
    #     self.canvas.grid(row=0, column=0, columnspan=3)
    #     ttk.Label(self.root, text="Force:").grid(row=1, column=0)
    #     self.force_slider.grid(row=1, column=1, sticky='ew')
    #     ttk.Button(self.root, text="Start", command=self.start_simulation).grid(row=2, column=0)
    #     # ttk.Button(self.root, text="Reset", command=self.reset).grid(row=2, column=1)
        
    def dynamics(self, s, a):
        angle, ang_vel = s
        force = np.clip(a, -self.force_max, self.force_max)
        ang_acc = (self.gravity/self.length)*np.sin(angle) + force/(self.mass*self.length**2)
        return ang_acc  # Remove [1][0] indexing

    def start_simulation(self):
        """Start the pendulum simulation"""
        self.running = True
        self.simulate()

    def reset(self):
        self.running = False
        self.state = np.array([0.0, 0.0])  # Reset to upright position
        self.draw_pendulum()

    # def simulate(self):
    #     if self.running:
    #         force = self.force_slider.get()
    #         ang_acc = self.dynamics(self.state, force)
    #         # Correct integration:
    #         self.state[1] += ang_acc * self.dt  # Update angular velocity first
    #         self.state[0] += self.state[1] * self.dt  # Then update angle
    #         self.draw_pendulum()
    #         self.root.after(int(self.dt * 1000), self.simulate)

    # def draw_pendulum(self):
    #     """Real-time visualization update"""
    #     self.canvas.delete("all")
    #     x = 200 + 100*self.length*np.sin(self.state[0])
    #     y = 200 - 100*self.length*np.cos(self.state[0])
    #     self.canvas.create_line(200, 200, x, y, width=4, fill='blue')
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='red')

# def run_gui():
#     root = tk.Tk()
#     InvertedPendulumGUI(root)
#     root.mainloop()

def generate_ssar(n_samples=100, dt=0.01, policy = None, vis = False):
    """Generate state, action, reward, next_state tuples"""
    length = 1.0
    mass = 1.0
    gravity = 9.81
    force_max = 1.

    states = []
    actions = []
    rewards = []
    next_states = []

    pendulum = InvertedPendulumGUI(None, length, mass, gravity, dt, force_max)
    
    for _ in range(int(n_samples)):
        if policy is None:
            action = np.array([[-0.1, 0,0, -0.1]]) @ np.sign(pendulum.state)
        else:
            action = policy.select_action(pendulum.state)
        pendulum.state[1] += pendulum.dynamics(pendulum.state, action) * dt
        pendulum.state[0] += pendulum.state[1] * dt
        if pendulum.state[0] > np.pi:
            pendulum.state[0] -= 2 * np.pi
        elif pendulum.state[0] < -np.pi:
            pendulum.state[0] += 2 * np.pi
        pendulum.state[1] = np.clip(pendulum.state[1], -10, 10)
        pendulum.state[0] = np.clip(pendulum.state[0], -np.pi, np.pi)
        # Store the state, action, reward, and next state
        
        states.append(pendulum.state.copy())
        actions.append(action)
        rewards.append(-pendulum.state[0]**2 - pendulum.state[1]**2 * 0.5)  # Placeholder for reward
        next_states.append(pendulum.state.copy())

    if vis:
        #  visualize the states
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(0, dt * n_samples, dt), np.array(states)[:, 0])
        ax.set_xlabel('Angle')
        ax.set_ylabel('Angular Velocity')
        ax.set_title('State Space')

        plt.show()
        
    

    return np.array(states),  np.array(next_states), np.array(actions), np.array(rewards),




if __name__ == "__main__":
    # run_gui()
    s,s1, a,r = generate_ssar(n_samples=1e4, dt=0.05, vis=True)

    
    ssar = np.concatenate((s, s1, a.reshape(-1, 1), r.reshape(-1, 1)), axis=1)
    print("States:", s.shape)
    print("Actions:", a.shape)
    print("Rewards:", r.shape)



    from utils import *
    from lspi.solvers import LSTDQSolver
    from lspi.train_offline import lspi_loop_offline
    samples = load_from_data(ssar, n_state=s.shape[1], n_action=1) 
    solver = LSTDQSolver()
    policy, all_policies = lspi_loop_offline(solver, samples, discount=0.8, epsilon=0.01, max_iterations=1)
    print("Policy weights:", policy.weights)

    generate_ssar(n_samples=1000, dt=0.05, policy=policy, vis=True)


    # testing the policy

    
