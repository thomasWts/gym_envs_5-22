from pathlib import Path
import argparse
import csv
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.cart_pole_final_env import CartPoleFinalEnv
from training.checkpoints import load_actor_critic


DEFAULT_MODEL_PATH = ROOT / "checkpoints" / "ppo_cart_pole_final_torch.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the final cart-pole model over multiple episodes.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="要评估的 checkpoint 路径。",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="评估 episode 数量。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="评估随机种子。",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="评估时打开 MuJoCo viewer。",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="可选：把每个 episode 的结果保存成 CSV。",
    )
    return parser.parse_args()


def run_episode(env, agent, device, seed=None, render=False):
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    episode_len = 0
    max_hold_steps = 0
    final_info = {}

    while True:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = agent.deterministic_action(obs_tensor)

        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])

        total_reward += reward
        episode_len += 1
        final_info = info
        max_hold_steps = max(max_hold_steps, info.get("center_hold_steps", 0))

        if render:
            time.sleep(env.model.opt.timestep * env.frame_skip)

        if terminated or truncated:
            break

    required_hold_steps = final_info.get("required_center_hold_steps", 1)
    hold_seconds = max_hold_steps * env.model.opt.timestep * env.frame_skip
    required_hold_seconds = required_hold_steps * env.model.opt.timestep * env.frame_skip

    return {
        "success": bool(final_info.get("success", False)),
        "return": float(total_reward),
        "length": int(episode_len),
        "max_hold_steps": int(max_hold_steps),
        "required_hold_steps": int(required_hold_steps),
        "max_hold_seconds": float(hold_seconds),
        "required_hold_seconds": float(required_hold_seconds),
        "final_x": float(final_info.get("x", np.nan)),
        "final_ball_x": float(final_info.get("ball_x", np.nan)),
        "final_theta_error": float(final_info.get("theta_error", np.nan)),
    }


def summarize(results):
    success_values = np.array([item["success"] for item in results], dtype=np.float32)
    returns = np.array([item["return"] for item in results], dtype=np.float32)
    lengths = np.array([item["length"] for item in results], dtype=np.float32)
    hold_seconds = np.array([item["max_hold_seconds"] for item in results], dtype=np.float32)
    final_ball_x = np.array([abs(item["final_ball_x"]) for item in results], dtype=np.float32)
    final_theta_error = np.array(
        [abs(item["final_theta_error"]) for item in results],
        dtype=np.float32,
    )

    return {
        "episodes": len(results),
        "success_rate": float(success_values.mean()),
        "avg_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "avg_length": float(lengths.mean()),
        "avg_max_hold_seconds": float(hold_seconds.mean()),
        "avg_abs_final_ball_x": float(final_ball_x.mean()),
        "avg_abs_final_theta_error": float(final_theta_error.mean()),
    }


def maybe_write_csv(path, results):
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}. "
            "Run: python scripts/train.py --task cart_pole_final"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CartPoleFinalEnv(render_mode="human" if args.render else None)
    agent, checkpoint = load_actor_critic(args.model, device=device)

    print("device:", device)
    print("model:", args.model)
    print("hidden_layers:", tuple(checkpoint.get("hidden_layers", (64, 64))))
    print("episodes:", args.episodes)
    print()

    results = []
    try:
        for episode in range(args.episodes):
            result = run_episode(
                env=env,
                agent=agent,
                device=device,
                seed=args.seed + episode,
                render=args.render,
            )
            results.append(result)
            print(
                f"episode {episode + 1:03d} | "
                f"success={result['success']} | "
                f"return={result['return']:9.2f} | "
                f"len={result['length']:4d} | "
                f"hold={result['max_hold_seconds']:.2f}/"
                f"{result['required_hold_seconds']:.2f}s | "
                f"ball_x={result['final_ball_x']:+.3f} | "
                f"theta_err={result['final_theta_error']:+.3f}"
            )
    finally:
        env.close()

    summary = summarize(results)
    print()
    print("Summary")
    print(f"success_rate:              {summary['success_rate']:.3f}")
    print(f"avg_return:                {summary['avg_return']:.2f}")
    print(f"std_return:                {summary['std_return']:.2f}")
    print(f"avg_length:                {summary['avg_length']:.1f}")
    print(f"avg_max_hold_seconds:      {summary['avg_max_hold_seconds']:.2f}")
    print(f"avg_abs_final_ball_x:      {summary['avg_abs_final_ball_x']:.3f}")
    print(f"avg_abs_final_theta_error: {summary['avg_abs_final_theta_error']:.3f}")

    maybe_write_csv(args.csv, results)
    if args.csv is not None:
        print(f"saved csv: {args.csv}")


if __name__ == "__main__":
    main()
