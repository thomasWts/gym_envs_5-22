from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


class FrankaPandaReachEnv(gym.Env):
    """Move the Franka Panda hand to a random 3D target."""

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
            model_path = (
                Path(__file__).resolve().parents[1]
                / "models"
                / "franka_emika_panda"
                / "reach_scene.xml"
            )

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.step_count = 0

        self.joint_names = tuple(f"joint{i}" for i in range(1, 8))
        self.actuator_names = tuple(f"actuator{i}" for i in range(1, 8))
        self.home_qpos = np.array(
            [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
            dtype=np.float64,
        )
        self.action_scale = np.array([0.035, 0.035, 0.035, 0.04, 0.045, 0.045, 0.05])
        self.command_qpos = self.home_qpos.copy()
        self.gripper_open_ctrl = 255.0
        self.prev_distance = 0.0
        self.target_pos = np.zeros(3, dtype=np.float64)

        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.actuator_names
        ]
        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "actuator8",
        )
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target_frame",
        )

        missing_joints = [name for name, joint_id in zip(self.joint_names, joint_ids) if joint_id < 0]
        missing_actuators = [
            name for name, actuator_id in zip(self.actuator_names, actuator_ids) if actuator_id < 0
        ]
        if missing_joints:
            raise ValueError(f"Missing joints in MuJoCo model: {missing_joints}")
        if missing_actuators:
            raise ValueError(f"Missing actuators in MuJoCo model: {missing_actuators}")
        if self.gripper_actuator_id < 0:
            raise ValueError("Actuator 'actuator8' not found in MuJoCo model.")
        if self.hand_body_id < 0:
            raise ValueError("Body 'hand' not found in MuJoCo model.")
        if self.target_body_id < 0:
            raise ValueError("Body 'target_frame' not found in MuJoCo model.")

        self.qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in joint_ids])
        self.dof_ids = np.array([self.model.jnt_dofadr[joint_id] for joint_id in joint_ids])
        self.actuator_ids = np.array(actuator_ids, dtype=np.int32)
        self.joint_low = np.array(
            [self.model.jnt_range[joint_id, 0] for joint_id in joint_ids],
            dtype=np.float64,
        )
        self.joint_high = np.array(
            [self.model.jnt_range[joint_id, 1] for joint_id in joint_ids],
            dtype=np.float64,
        )

        obs_dim = 7 + 7 + 3 + 3 + 3 + 3 + 7
        obs_high = np.array([10.0] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.command_qpos = np.clip(
            self.home_qpos + self.np_random.uniform(-0.08, 0.08, size=7),
            self.joint_low,
            self.joint_high,
        )
        self.data.qpos[self.qpos_ids] = self.command_qpos
        self.data.qvel[self.dof_ids] = self.np_random.uniform(-0.01, 0.01, size=7)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.actuator_ids] = self.command_qpos
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_open_ctrl

        mujoco.mj_forward(self.model, self.data)
        self.target_pos = self._sample_target()
        self._sync_target_frame()
        mujoco.mj_forward(self.model, self.data)

        self.prev_distance = self._distance()
        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info(success=False)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.command_qpos = np.clip(
            self.command_qpos + self.action_scale * action.astype(np.float64),
            self.joint_low,
            self.joint_high,
        )
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.actuator_ids] = self.command_qpos
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_open_ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        hand_pos = self._hand_pos()
        hand_vel = self._hand_vel()
        error = self.target_pos - hand_pos
        distance = float(np.linalg.norm(error))
        speed = float(np.linalg.norm(hand_vel))
        progress = self.prev_distance - distance
        self.prev_distance = distance

        qpos = self.data.qpos[self.qpos_ids]
        limit_margin = np.minimum(qpos - self.joint_low, self.joint_high - qpos)
        near_limit = float(np.mean(limit_margin < 0.12))

        reward = 1.0
        reward += 16.0 * progress
        reward -= 4.5 * distance
        reward -= 0.04 * float(np.dot(action, action))
        reward -= 0.03 * speed
        reward -= 0.3 * near_limit

        success = distance < 0.06 and speed < 0.45
        if distance < 0.15:
            reward += 3.0 * (1.0 - distance / 0.15)
        if success:
            reward += 300.0

        bad_height = hand_pos[2] < 0.06
        if bad_height:
            reward -= 25.0

        terminated = bool(success or bad_height)
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = self._get_info(success=success)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = np.array([0.35, 0.0, 0.45], dtype=np.float64)
            self.viewer.cam.distance = 1.8
            self.viewer.cam.azimuth = 135.0
            self.viewer.cam.elevation = -25.0

        if self.viewer.is_running():
            self.viewer.sync()

        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _sample_target(self):
        return np.array(
            [
                self.np_random.uniform(0.30, 0.62),
                self.np_random.uniform(-0.32, 0.32),
                self.np_random.uniform(0.22, 0.62),
            ],
            dtype=np.float64,
        )

    def _hand_pos(self):
        return self.data.xpos[self.hand_body_id].copy()

    def _hand_vel(self):
        return self.data.cvel[self.hand_body_id, 3:6].copy()

    def _distance(self):
        return float(np.linalg.norm(self.target_pos - self._hand_pos()))

    def _get_obs(self):
        qpos = self.data.qpos[self.qpos_ids].copy()
        qvel = self.data.qvel[self.dof_ids].copy()
        hand_pos = self._hand_pos()
        hand_vel = self._hand_vel()
        error = self.target_pos - hand_pos

        qpos_center = 0.5 * (self.joint_low + self.joint_high)
        qpos_half_span = 0.5 * (self.joint_high - self.joint_low)

        obs = np.concatenate(
            [
                (qpos - qpos_center) / qpos_half_span,
                qvel / 3.0,
                hand_pos / 0.8,
                hand_vel / 2.0,
                self.target_pos / 0.8,
                error / 0.8,
                self.command_qpos / np.pi,
            ]
        )
        return obs.astype(np.float32)

    def _get_info(self, success: bool):
        hand_pos = self._hand_pos()
        hand_vel = self._hand_vel()
        error = self.target_pos - hand_pos

        return {
            "hand_position": hand_pos,
            "hand_velocity": hand_vel,
            "target_position": self.target_pos.copy(),
            "error": error,
            "distance": float(np.linalg.norm(error)),
            "speed": float(np.linalg.norm(hand_vel)),
            "success": bool(success),
            "contacts": int(self.data.ncon),
        }

    def _sync_target_frame(self):
        self.model.body_pos[self.target_body_id] = self.target_pos
