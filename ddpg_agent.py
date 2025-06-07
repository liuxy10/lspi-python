import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=20):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=20):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, state, action, reward, next_state, done):
        # Convert to numpy arrays immediately
        self.buffer.append(self.Transition(
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        ))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Stack each component separately
        batch = self.Transition(
            np.stack([t.state for t in samples]),
            np.stack([t.action for t in samples]),
            np.stack([t.reward for t in samples]),
            np.stack([t.next_state for t in samples]),
            np.stack([t.done for t in samples])
        )
        return batch


# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = ReplayBuffer(1000)
        self.tau = 0.005
        self.gamma = 0.99
        
        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def get_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        action += noise_scale * np.random.randn(len(action))
        return np.clip(action, -1, 1)
    
    def update(self, batch_size=128):
        if len(self.replay_buffer.buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(np.array(batch.state)).to(device)
        action = torch.FloatTensor(np.array(batch.action)).to(device)
        reward = torch.FloatTensor(np.array(batch.reward)).to(device)
        next_state = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done = torch.FloatTensor(np.array(batch.done)).to(device)
        
        # Critic update
        with torch.no_grad():
            target_actions = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, target_actions)
            target_Q = reward + (1 - done) * self.gamma * target_Q
            
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update targets
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Usage with your emulator
