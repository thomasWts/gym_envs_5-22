import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import ActorCritic
from envs.cart_reach_env import CartReachEnv


MODEL_PATH = ROOT / "checkpoints" / "ppo_cart_reach_torch.pt"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run train_cart_reach_ppo.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CartReachEnv(render_mode="human")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    agent = ActorCritic(
        obs_dim=checkpoint["obs_dim"],
        act_dim=checkpoint["act_dim"],
    ).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    obs, _ = env.reset(seed=1)

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
                    f"error={info['error']:.3f}"
                )
                obs, _ = env.reset()

            time.sleep(env.model.opt.timestep * env.frame_skip)
    finally:
        env.close()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
