import gymnasium
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super(Actor, self).__init__()
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=DEVICE)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=DEVICE)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )
        self.mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def get_raw(self, state):
        x = self.net(state)
        mean_raw = self.mean(x)
        log_std_clamped = self.log_std.clamp(min=-20, max=2)
        std = torch.exp(log_std_clamped)
        return Normal(mean_raw, std)

    def get_action(self, state):
        raw_dist = self.get_raw(state)
        mean_raw_action = raw_dist.mean
        mean_action = torch.tanh(mean_raw_action)
        action = mean_action * self.action_scale + self.action_bias
        final_action = torch.clamp(action, self.action_low, self.action_high)
        return final_action

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self, load_path="model"):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (21,), np.float64)
        state_dim = 67
        action_dim = 21
        action_low = self.action_space.low
        action_high = self.action_space.high
        self.actor = Actor(state_dim, action_dim, action_low, action_high).to(DEVICE)
        self.load(load_path)
        self.actor.eval()

    def load(self, load_path):
        checkpoint = torch.load(load_path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        print(f"Loaded Actor model from {load_path}")

    def act(self, observation):
        state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.get_action(state)
        return action.cpu().numpy().flatten()
