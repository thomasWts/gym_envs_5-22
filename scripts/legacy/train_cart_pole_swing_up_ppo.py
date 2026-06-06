from pathlib import Path
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import ActorCritic
from envs.cart_pole_swing_up_env import CartPoleSwingUpEnv


MODEL_PATH = ROOT / "checkpoints" / "ppo_cart_pole_swing_up_torch.pt"
DEFAULT_INIT_PATH = ROOT / "checkpoints" / "ppo_cart_pole_wide_recover_torch.pt"
HIDDEN_LAYERS = (128, 128, 128)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CartPoleSwingUpEnv with PPO.",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=DEFAULT_INIT_PATH,
        help=(
            "用已有 checkpoint 初始化策略。默认使用 wide recover 难度的 checkpoint。"
        ),
    )
    parser.add_argument(
        "--no-init-from",
        action="store_true",
        help="不加载旧 checkpoint，完全从随机初始化开始训练。",
    )
    return parser.parse_args()


def load_checkpoint_if_available(agent, checkpoint_path, device, obs_dim, act_dim):
    if checkpoint_path is None:
        print("init_from: disabled, training from random initialization")
        return

    if not checkpoint_path.exists():
        print(f"init_from: {checkpoint_path} not found, training from random initialization")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("obs_dim") != obs_dim or checkpoint.get("act_dim") != act_dim:
        raise ValueError(
            "Checkpoint dimension mismatch: "
            f"checkpoint obs_dim={checkpoint.get('obs_dim')}, act_dim={checkpoint.get('act_dim')} | "
            f"env obs_dim={obs_dim}, act_dim={act_dim}"
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


def main():
    args = parse_args()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    seed = 3
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = CartPoleSwingUpEnv(render_mode=None)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    agent = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layers=HIDDEN_LAYERS,
    ).to(device)
    init_from = None if args.no_init_from else args.init_from
    load_checkpoint_if_available(
        agent=agent,
        checkpoint_path=init_from,
        device=device,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    total_updates = 300
    steps_per_update = 2048
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    train_epochs = 10
    minibatch_size = 256
    value_coef = 0.5
    entropy_coef = 0.008
    max_grad_norm = 0.5

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_len = 0
    recent_returns = []
    recent_lengths = []
    recent_successes = []

    for update in range(1, total_updates + 1):
        obs_buf = []
        raw_action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        for _ in range(steps_per_update):
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
                recent_successes.append(float(info["success"]))
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
            gamma=gamma,
            gae_lambda=gae_lambda,
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

        indices = np.arange(steps_per_update)
        last_loss_info = None
        for _ in range(train_epochs):
            np.random.shuffle(indices)

            for start in range(0, steps_per_update, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                loss, loss_info = compute_ppo_loss(
                    agent=agent,
                    obs=obs_tensor[mb_idx],
                    raw_action=raw_action_tensor[mb_idx],
                    old_logprob=old_logprob_tensor[mb_idx],
                    advantage=advantage_tensor[mb_idx],
                    target_return=return_tensor[mb_idx],
                    clip_ratio=clip_ratio,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                )
                last_loss_info = loss_info

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
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
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "hidden_layers": HIDDEN_LAYERS,
                },
                MODEL_PATH,
            )
            print(f"saved checkpoint to {MODEL_PATH}")

    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_layers": HIDDEN_LAYERS,
        },
        MODEL_PATH,
    )
    env.close()
    print(f"saved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
