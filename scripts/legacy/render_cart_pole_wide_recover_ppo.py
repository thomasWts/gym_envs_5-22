import time
from pathlib import Path
import argparse
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import ActorCritic
from envs.cart_pole_wide_recover_env import CartPoleWideRecoverEnv


MODEL_PATH = ROOT / "checkpoints" / "ppo_cart_pole_wide_recover_torch.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render CartPoleWideRecoverEnv policy.",
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
            "Run scripts/train_cart_pole_wide_recover_ppo.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CartPoleWideRecoverEnv(render_mode="human")

    checkpoint = torch.load(model_path, map_location=device)
    agent = ActorCritic(
        obs_dim=checkpoint["obs_dim"],
        act_dim=checkpoint["act_dim"],
    ).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    obs, _ = env.reset(seed=3)

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
                    f"theta_error={info['theta_error']:.3f}"
                )
                obs, _ = env.reset()

            time.sleep(env.model.opt.timestep * env.frame_skip)
    finally:
        env.close()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
