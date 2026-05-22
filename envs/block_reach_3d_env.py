from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


class BlockReach3DEnv(gym.Env):
    """Move a 3D block to a random target position and stop there."""

    metadata = {
        "render_modes": ["human"],
        "render_fps": 100,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | Path | None = None,
        max_episode_steps: int = 250,
        frame_skip: int = 5,
    ):
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "models" / "block_reach_3d.xml"

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.force_limit = 8.0
        self.xyz_limit = np.array([0.78, 0.78, 0.83], dtype=np.float64)
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.prev_distance = 0.0
        self.step_count = 0

        self.success_distance = 0.07
        self.success_speed = 0.25
        self.slow_down_distance = 0.18

        joint_names = ("block_slide_x", "block_slide_y", "block_slide_z")
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ]
        self.target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "target_site",
        )

        missing_joints = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
        if missing_joints:
            raise ValueError(f"Missing joints in MuJoCo model: {missing_joints}")
        if self.target_site_id < 0:
            raise ValueError("Site 'target_site' not found in MuJoCo model.")

        self.qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in joint_ids])
        self.dof_ids = np.array([self.model.jnt_dofadr[joint_id] for joint_id in joint_ids])

        self.observation_space = spaces.Box(
            low=np.array([-2.0] * 3 + [-10.0] * 3 + [-2.0] * 3 + [-2.0] * 3, dtype=np.float32),
            high=np.array([2.0] * 3 + [10.0] * 3 + [2.0] * 3 + [2.0] * 3, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        pos = self._sample_position()
        target_pos = self._sample_position()
        for _ in range(100):
            if np.linalg.norm(target_pos - pos) >= 0.25:
                break
            target_pos = self._sample_position()

        self.target_pos = target_pos
        self.prev_distance = float(np.linalg.norm(self.target_pos - pos))
        self.step_count = 0

        self.data.qpos[self.qpos_ids] = pos
        self.data.qvel[self.dof_ids] = self.np_random.uniform(-0.05, 0.05, size=3)
        self.data.ctrl[:] = 0.0
        self._sync_target_site()

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info(force=np.zeros(3), success=False)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        force = self.force_limit * action.astype(np.float64)
        self.data.ctrl[:] = force

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        pos = self._position()
        vel = self._velocity()
        error = self.target_pos - pos
        distance = float(np.linalg.norm(error))
        speed = float(np.linalg.norm(vel))
        progress = self.prev_distance - distance
        self.prev_distance = distance

        reward = 1.0
        reward += 3.0 * progress
        reward -= 3.0 * distance
        # reward -= 4.0 * distance**2
        reward -= 0.10 * speed**2
        reward -= 0.0015 * float(np.dot(force, force))

        if distance < self.slow_down_distance:
            near_ratio = 1.0 - distance / self.slow_down_distance
            reward += 2.0 * near_ratio
            reward -= 0.60 * speed**2

        success = distance < self.success_distance and speed < self.success_speed
        out_of_bounds = (
            abs(pos[0]) > self.xyz_limit[0]
            or abs(pos[1]) > self.xyz_limit[1]
            or pos[2] < 0.10
            or pos[2] > self.xyz_limit[2]
        )

        terminated = False
        if success:
            reward += 200.0
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

    def _sample_position(self):
        return np.array(
            [
                self.np_random.uniform(-0.55, 0.55),
                self.np_random.uniform(-0.55, 0.55),
                self.np_random.uniform(0.18, 0.72),
            ],
            dtype=np.float64,
        )

    def _position(self):
        return self.data.qpos[self.qpos_ids].copy()

    def _velocity(self):
        return self.data.qvel[self.dof_ids].copy()

    def _get_obs(self):
        pos = self._position()
        vel = self._velocity()
        error = self.target_pos - pos

        obs = np.concatenate(
            [
                pos / 0.8,
                vel / 3.0,
                self.target_pos / 0.8,
                error / 0.8,
            ]
        )
        return obs.astype(np.float32)

    def _get_info(self, force, success: bool):
        pos = self._position()
        vel = self._velocity()
        error = self.target_pos - pos

        return {
            "position": pos,
            "velocity": vel,
            "target_position": self.target_pos.copy(),
            "error": error,
            "distance": float(np.linalg.norm(error)),
            "speed": float(np.linalg.norm(vel)),
            "force": np.asarray(force, dtype=np.float64),
            "success": bool(success),
        }

    def _sync_target_site(self):
        self.model.site_pos[self.target_site_id] = self.target_pos
