import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def distribution(self, obs):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs):
        dist = self.distribution(obs)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        value = self.value(obs)
        return raw_action, action, log_prob, value

    def deterministic_action(self, obs):
        mean = self.actor(obs)
        return torch.tanh(mean)

    def evaluate_actions(self, obs, raw_action):
        dist = self.distribution(obs)
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value

    @staticmethod
    def _squashed_log_prob(dist, raw_action, action):
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return log_prob - correction
