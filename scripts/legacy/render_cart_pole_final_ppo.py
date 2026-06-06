import time
from pathlib import Path
import argparse
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.cart_pole_final_env import CartPoleFinalEnv
from training.checkpoints import load_actor_critic


MODEL_PATH = ROOT / "checkpoints" / "ppo_cart_pole_final_torch.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render CartPoleFinalEnv policy.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="要加载的 checkpoint 路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run scripts/train_cart_pole_final_ppo.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CartPoleFinalEnv(render_mode="human")

    agent, _ = load_actor_critic(model_path, device=device)

    obs, _ = env.reset(seed=5)

    try:
        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action = agent.deterministic_action(obs_tensor)

            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])

            if terminated or truncated:
                print(
                    "episode done | "
                    f"success={info['success']} | "
                    f"x={info['x']:.3f} | "
                    f"ball_x={info['ball_x']:.3f} | "
                    f"theta_error={info['theta_error']:.3f} | "
                    f"hold={info['center_hold_steps']}/{info['required_center_hold_steps']}"
                )
                obs, _ = env.reset()

            time.sleep(env.model.opt.timestep * env.frame_skip)
    finally:
        env.close()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
