import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def build_mlp(input_dim, output_dim, hidden_layers):
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class SACActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers=(256, 256)):
        super().__init__()
        self.hidden_layers = tuple(hidden_layers)
        self.trunk = build_mlp(obs_dim, hidden_layers[-1], hidden_layers[:-1])
        self.mean = nn.Linear(hidden_layers[-1], act_dim)
        self.log_std = nn.Linear(hidden_layers[-1], act_dim)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, obs):
        mean, _ = self(obs)
        return torch.tanh(mean)


class SACCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers=(256, 256)):
        super().__init__()
        input_dim = obs_dim + act_dim
        self.q1 = build_mlp(input_dim, 1, hidden_layers)
        self.q2 = build_mlp(input_dim, 1, hidden_layers)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())
