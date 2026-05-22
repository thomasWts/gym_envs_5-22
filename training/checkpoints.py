from pathlib import Path

import torch

from agents.ppo_agent import ActorCritic


def load_actor_critic(checkpoint_path: str | Path, device):
    """
    从 checkpoint 加载 ActorCritic。

    checkpoint 里至少需要：
        model_state_dict
        obs_dim
        act_dim

    如果 checkpoint 里有 hidden_layers，就按保存时的网络结构恢复；
    如果没有，就兼容旧模型，默认使用 (64, 64)。
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    hidden_layers = tuple(checkpoint.get("hidden_layers", (64, 64)))
    agent = ActorCritic(
        obs_dim=checkpoint["obs_dim"],
        act_dim=checkpoint["act_dim"],
        hidden_layers=hidden_layers,
    ).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    return agent, checkpoint

