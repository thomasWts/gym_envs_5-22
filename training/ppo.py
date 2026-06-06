from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agents.ppo_agent import ActorCritic
from training.configs import PPOTrainConfig


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_ppo_loss(
    agent,
    obs,
    raw_action,
    old_logprob,
    advantage,
    target_return,
    clip_ratio,
    value_coef,
    entropy_coef,
):
    new_logprob, entropy, value = agent.evaluate_actions(obs, raw_action)

    ratio = torch.exp(new_logprob - old_logprob)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    policy_loss = -torch.min(
        ratio * advantage,
        clipped_ratio * advantage,
    ).mean()
    value_loss = (value - target_return).pow(2).mean()
    entropy_loss = entropy.mean()
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

    loss_info = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy_loss.detach(),
    }
    return total_loss, loss_info


def load_compatible_checkpoint(agent, checkpoint_path, device, obs_dim, act_dim):
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

    current_state = agent.state_dict()
    pretrained_state = checkpoint["model_state_dict"]
    compatible_state = {
        name: value
        for name, value in pretrained_state.items()
        if name in current_state and current_state[name].shape == value.shape
    }

    current_state.update(compatible_state)
    agent.load_state_dict(current_state)
    print(
        f"init_from: loaded {len(compatible_state)}/{len(current_state)} "
        f"compatible tensors from {checkpoint_path}"
    )


def save_checkpoint(path, agent, obs_dim, act_dim, hidden_layers):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_layers": tuple(hidden_layers),
        },
        path,
    )


def train_ppo(
    config: PPOTrainConfig,
    root: Path,
    init_from: Path | None = None,
    no_init_from: bool = False,
    total_updates: int | None = None,
):
    seed = config.seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("task:", config.name)
    print("device:", device)

    env = config.env_cls(render_mode=None)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    agent = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layers=config.hidden_layers,
    ).to(device)

    if no_init_from:
        resolved_init_from = None
    elif init_from is not None:
        resolved_init_from = init_from
    else:
        resolved_init_from = config.init_checkpoint_path(root)

    load_compatible_checkpoint(
        agent=agent,
        checkpoint_path=resolved_init_from,
        device=device,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    model_path = config.checkpoint_path(root)
    update_count = total_updates or config.total_updates

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_len = 0
    recent_returns = []
    recent_lengths = []
    recent_successes = []

    for update in range(1, update_count + 1):
        obs_buf = []
        raw_action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        for _ in range(config.steps_per_update):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                raw_action, action, log_prob, value = agent.act(obs_tensor)

            next_obs, reward, terminated, truncated, info = env.step(
                action.cpu().numpy()[0]
            )
            done = terminated or truncated

            obs_buf.append(obs)
            raw_action_buf.append(raw_action.cpu().numpy()[0])
            logprob_buf.append(log_prob.cpu().numpy()[0])
            reward_buf.append(reward)
            done_buf.append(done)
            value_buf.append(value.cpu().numpy()[0])

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

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = agent.value(obs_tensor).cpu().numpy()[0]

        rewards = np.asarray(reward_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)
        values = np.asarray(value_buf, dtype=np.float32)
        advantages, returns = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
        raw_action_tensor = torch.as_tensor(
            np.asarray(raw_action_buf),
            dtype=torch.float32,
            device=device,
        )
        old_logprob_tensor = torch.as_tensor(
            np.asarray(logprob_buf),
            dtype=torch.float32,
            device=device,
        )
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

        indices = np.arange(config.steps_per_update)
        last_loss_info = None
        for _ in range(config.train_epochs):
            np.random.shuffle(indices)

            for start in range(0, config.steps_per_update, config.minibatch_size):
                mb_idx = indices[start : start + config.minibatch_size]
                loss, loss_info = compute_ppo_loss(
                    agent=agent,
                    obs=obs_tensor[mb_idx],
                    raw_action=raw_action_tensor[mb_idx],
                    old_logprob=old_logprob_tensor[mb_idx],
                    advantage=advantage_tensor[mb_idx],
                    target_return=return_tensor[mb_idx],
                    clip_ratio=config.clip_ratio,
                    value_coef=config.value_coef,
                    entropy_coef=config.entropy_coef,
                )
                last_loss_info = loss_info

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

        mean_return = np.mean(recent_returns[-20:]) if recent_returns else 0.0
        mean_len = np.mean(recent_lengths[-20:]) if recent_lengths else 0.0
        success_rate = np.mean(recent_successes[-20:]) if recent_successes else 0.0
        policy_loss = last_loss_info["policy_loss"].item() if last_loss_info else 0.0
        value_loss = last_loss_info["value_loss"].item() if last_loss_info else 0.0
        entropy = last_loss_info["entropy"].item() if last_loss_info else 0.0

        print(
            f"update {update:03d} | "
            f"mean_return={mean_return:8.2f} | "
            f"mean_len={mean_len:6.1f} | "
            f"success_rate={success_rate:5.2f} | "
            f"pi_loss={policy_loss:7.3f} | "
            f"v_loss={value_loss:8.3f} | "
            f"entropy={entropy:6.3f} | "
            f"log_std={agent.log_std.detach().cpu().numpy()}"
        )

        if update % 20 == 0:
            save_checkpoint(
                model_path,
                agent=agent,
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_layers=config.hidden_layers,
            )
            print(f"saved checkpoint to {model_path}")

    save_checkpoint(
        model_path,
        agent=agent,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layers=config.hidden_layers,
    )
    env.close()
    print(f"saved final model to {model_path}")
