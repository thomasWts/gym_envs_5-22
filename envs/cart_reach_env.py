from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


class CartReachEnv(gym.Env):
    """Move a 1D MuJoCo cart to a random target and stop there."""

    metadata = {
        "render_modes": ["human"],
        "render_fps": 100,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | Path | None = None,
        max_episode_steps: int = 200,
        frame_skip: int = 5,
    ):
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "models" / "cart_reach.xml"

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.force_limit = 10.0
        self.x_limit = 0.78
        self.target_x = 0.0
        self.step_count = 0

        self.cart_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "cart_slide_x",
        )
        self.target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "target_site",
        )

        if self.cart_joint_id < 0:
            raise ValueError("Joint 'cart_slide_x' not found in MuJoCo model.")
        if self.target_site_id < 0:
            raise ValueError("Site 'target_site' not found in MuJoCo model.")

        self.cart_qpos_id = self.model.jnt_qposadr[self.cart_joint_id]
        self.cart_dof_id = self.model.jnt_dofadr[self.cart_joint_id]

        self.observation_space = spaces.Box(
            low=np.array([-2.0, -10.0, -1.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 10.0, 1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        x = self.np_random.uniform(-0.45, 0.45)
        target_x = self.np_random.uniform(-0.65, 0.65)

        if abs(target_x - x) < 0.20:
            target_x = np.clip(target_x + np.sign(target_x - x + 1e-6) * 0.25, -0.65, 0.65)

        self.target_x = float(target_x)
        self.step_count = 0

        self.data.qpos[self.cart_qpos_id] = x
        self.data.qvel[self.cart_dof_id] = self.np_random.uniform(-0.05, 0.05)
        self.data.ctrl[0] = 0.0
        self._sync_target_site()

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info(force=0.0, success=False)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        force = self.force_limit * float(action[0])
        self.data.ctrl[0] = force

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        x = float(self.data.qpos[self.cart_qpos_id])
        x_dot = float(self.data.qvel[self.cart_dof_id])
        error = self.target_x - x

        reward = 1.0
        reward -= 8.0 * error**2
        reward -= 0.10 * x_dot**2
        reward -= 0.001 * force**2

        success = abs(error) < 0.03 and abs(x_dot) < 0.20
        out_of_bounds = abs(x) > self.x_limit

        terminated = False
        if success:
            reward += 10.0
            terminated = True
        elif out_of_bounds:
            reward -= 20.0
            terminated = True

        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = self._get_info(force=force, success=success)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        if self.viewer.is_running():
            self.viewer.sync()

        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        x = float(self.data.qpos[self.cart_qpos_id])
        x_dot = float(self.data.qvel[self.cart_dof_id])
        error = self.target_x - x

        return np.array(
            [
                x / 0.8,
                x_dot / 3.0,
                self.target_x / 0.8,
                error / 0.8,
            ],
            dtype=np.float32,
        )

    def _get_info(self, force: float, success: bool):
        x = float(self.data.qpos[self.cart_qpos_id])
        x_dot = float(self.data.qvel[self.cart_dof_id])
        error = self.target_x - x

        return {
            "x": x,
            "x_dot": x_dot,
            "target_x": self.target_x,
            "error": error,
            "force": float(force),
            "success": bool(success),
        }

    def _sync_target_site(self):
        self.model.site_pos[self.target_site_id, 0] = self.target_x
