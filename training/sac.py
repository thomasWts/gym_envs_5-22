from collections import deque
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from agents.sac_agent import SACActor, SACCritic, hard_update, soft_update
from training.sac_configs import SACTrainConfig


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "action": torch.as_tensor(self.action[idx], dtype=torch.float32, device=device),
            "reward": torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "done": torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_sac_checkpoint(path, actor, critic, obs_dim, act_dim, hidden_layers):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_layers": tuple(hidden_layers),
        },
        path,
    )


def load_sac_actor(checkpoint_path: str | Path, device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_layers = tuple(checkpoint.get("hidden_layers", (256, 256)))
    actor = SACActor(
        obs_dim=checkpoint["obs_dim"],
        act_dim=checkpoint["act_dim"],
        hidden_layers=hidden_layers,
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor, checkpoint


def load_compatible_sac_checkpoint(
    actor,
    critic,
    checkpoint_path,
    device,
    obs_dim,
    act_dim,
):
    if checkpoint_path is None:
        print("init_from: disabled, training from random initialization")
        return

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"init_from: {checkpoint_path} not found, training from random initialization")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("act_dim") != act_dim:
        raise ValueError(
            "Checkpoint dimension mismatch: "
            f"checkpoint obs_dim={checkpoint.get('obs_dim')}, act_dim={checkpoint.get('act_dim')} | "
            f"env obs_dim={obs_dim}, act_dim={act_dim}"
        )
    if checkpoint.get("obs_dim") != obs_dim:
        print(
            "init_from: obs_dim differs, loading only compatible tensors | "
            f"checkpoint obs_dim={checkpoint.get('obs_dim')} -> env obs_dim={obs_dim}"
        )

    _load_compatible_state(
        module=actor,
        pretrained_state=checkpoint["actor_state_dict"],
        module_name="actor",
        checkpoint_path=checkpoint_path,
    )
    _load_compatible_state(
        module=critic,
        pretrained_state=checkpoint["critic_state_dict"],
        module_name="critic",
        checkpoint_path=checkpoint_path,
    )


def _load_compatible_state(module, pretrained_state, module_name, checkpoint_path):
    current_state = module.state_dict()
    compatible_state = {
        name: value
        for name, value in pretrained_state.items()
        if name in current_state and current_state[name].shape == value.shape
    }

    current_state.update(compatible_state)
    module.load_state_dict(current_state)
    print(
        f"init_from: loaded {module_name} {len(compatible_state)}/{len(current_state)} "
        f"compatible tensors from {checkpoint_path}"
    )


def train_sac(
    config: SACTrainConfig,
    root: Path,
    total_steps: int | None = None,
    init_from: Path | None = None,
    no_init_from: bool = False,
):
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = config.env_cls(render_mode=None)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    actor = SACActor(obs_dim, act_dim, config.hidden_layers).to(device)
    critic = SACCritic(obs_dim, act_dim, config.hidden_layers).to(device)
    critic_target = SACCritic(obs_dim, act_dim, config.hidden_layers).to(device)

    if no_init_from:
        resolved_init_from = None
    elif init_from is not None:
        resolved_init_from = init_from
    else:
        resolved_init_from = config.init_checkpoint_path(root)

    load_compatible_sac_checkpoint(
        actor=actor,
        critic=critic,
        checkpoint_path=resolved_init_from,
        device=device,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )
    hard_update(critic_target, critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)

    if config.alpha is None:
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.learning_rate)
        target_entropy = -float(act_dim)
    else:
        log_alpha = None
        alpha_optimizer = None
        target_entropy = None

    replay = ReplayBuffer(obs_dim, act_dim, config.replay_size)
    model_path = config.checkpoint_path(root)
    step_count = total_steps or config.total_steps

    obs, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_len = 0
    recent_returns = deque(maxlen=20)
    recent_lengths = deque(maxlen=20)
    recent_successes = deque(maxlen=20)

    print("task:", config.name)
    print("algo: SAC")
    print("device:", device)

    for step in range(1, step_count + 1):
        if step <= config.start_steps:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.sample(obs_tensor)
            action = action.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, float(terminated))

        episode_return += reward
        episode_len += 1
        obs = next_obs

        if done:
            recent_returns.append(episode_return)
            recent_lengths.append(episode_len)
            recent_successes.append(float(info.get("success", False)))
            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        if step >= config.update_after and step % config.update_every == 0:
            for _ in range(config.update_every):
                batch = replay.sample(config.batch_size, device)

                if config.alpha is None:
                    alpha = log_alpha.exp()
                else:
                    alpha = torch.tensor(config.alpha, dtype=torch.float32, device=device)

                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(batch["next_obs"])
                    target_q1, target_q2 = critic_target(batch["next_obs"], next_action)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    backup = batch["reward"] + config.gamma * (1.0 - batch["done"]) * target_q

                q1, q2 = critic(batch["obs"], batch["action"])
                critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                new_action, log_prob = actor.sample(batch["obs"])
                q1_pi, q2_pi = critic(batch["obs"], new_action)
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha.detach() * log_prob - q_pi).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if config.alpha is None:
                    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                soft_update(critic_target, critic, config.tau)

        if step % 1000 == 0:
            mean_return = np.mean(recent_returns) if recent_returns else 0.0
            mean_len = np.mean(recent_lengths) if recent_lengths else 0.0
            success_rate = np.mean(recent_successes) if recent_successes else 0.0
            alpha_value = log_alpha.exp().item() if config.alpha is None else config.alpha
            print(
                f"step {step:07d} | "
                f"mean_return={mean_return:8.2f} | "
                f"mean_len={mean_len:6.1f} | "
                f"success_rate={success_rate:5.2f} | "
                f"buffer={replay.size:7d} | "
                f"alpha={alpha_value:.4f}"
            )

        if step % config.save_every == 0:
            save_sac_checkpoint(
                model_path,
                actor=actor,
                critic=critic,
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_layers=config.hidden_layers,
            )

    save_sac_checkpoint(
        model_path,
        actor=actor,
        critic=critic,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layers=config.hidden_layers,
    )
    env.close()
    print(f"saved final SAC model to {model_path}")
