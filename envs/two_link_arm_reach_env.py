from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


class TwoLinkArmReachEnv(gym.Env):
    """Control a planar two-link arm to move its end effector to a target."""

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
                / "two_link_arm_reach.xml"
            )

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.torque_limit = 3.0
        self.link_reach = 0.78
        self.success_distance = 0.09
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.previous_distance = 0.0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        self.shoulder_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "shoulder",
        )
        self.elbow_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "elbow",
        )
        self.ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "end_effector",
        )
        self.target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "target_site",
        )

        for name, item_id in (
            ("shoulder", self.shoulder_joint_id),
            ("elbow", self.elbow_joint_id),
            ("end_effector", self.ee_site_id),
            ("target_site", self.target_site_id),
        ):
            if item_id < 0:
                raise ValueError(f"{name!r} not found in MuJoCo model.")

        self.shoulder_qpos_id = self.model.jnt_qposadr[self.shoulder_joint_id]
        self.elbow_qpos_id = self.model.jnt_qposadr[self.elbow_joint_id]
        self.shoulder_dof_id = self.model.jnt_dofadr[self.shoulder_joint_id]
        self.elbow_dof_id = self.model.jnt_dofadr[self.elbow_joint_id]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        q1, q2 = self._sample_initial_qpos()

        self.data.qpos[self.shoulder_qpos_id] = q1
        self.data.qpos[self.elbow_qpos_id] = q2
        self.data.qvel[self.shoulder_dof_id] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[self.elbow_dof_id] = self.np_random.uniform(-0.05, 0.05)
        self.data.ctrl[:] = 0.0

        self.target_pos = self._sample_target()
        self._sync_target_site()
        self.step_count = 0
        self.previous_action[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.previous_distance = self._distance_to_target()

        obs = self._get_obs()
        info = self._get_info(torque=np.zeros(2, dtype=np.float32), success=False)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        torque = self.torque_limit * action
        self.data.ctrl[:] = torque

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        ee_pos = self._end_effector_pos()
        ee_vel = self._end_effector_vel()
        error = self.target_pos - ee_pos
        distance = float(np.linalg.norm(error))
        progress = self.previous_distance - distance
        ee_speed = float(np.linalg.norm(ee_vel))
        action_delta = action - self.previous_action

        # 训练版 reward：
        # 1. 距离越小越好；
        # 2. 比上一帧更接近目标就奖励；
        # 3. 成功圈外一直吃距离惩罚，避免停在目标附近刷分；
        # 4. 目标附近速度和动作变化要小，减少来回甩；
        # 5. 超时失败给额外惩罚，让“差一点但没成功”和“成功”差距更明确。
        reward = -10.0 * distance
        reward += 20.0 * progress
        reward -= 0.001 * float(np.sum(torque**2))

        if distance < 0.20:
            reward -= 0.10 * ee_speed
            reward -= 0.03 * float(np.sum(action_delta**2))

        near_success_but_stuck = (
            self.success_distance <= distance < self.success_distance + 0.04
            and progress < 0.001
        )
        if near_success_but_stuck:
            reward -= 0.30

        self.previous_distance = distance
        self.previous_action = action.copy()

        success = distance < self.success_distance
        terminated = False
        if success:
            reward += 30.0
            terminated = True

        truncated = self.step_count >= self.max_episode_steps
        if truncated and not success:
            reward -= 10.0

        obs = self._get_obs()
        info = self._get_info(torque=torque, success=success)

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
        q1 = float(self.data.qpos[self.shoulder_qpos_id])
        q2 = float(self.data.qpos[self.elbow_qpos_id])
        qvel = np.array(
            [
                self.data.qvel[self.shoulder_dof_id],
                self.data.qvel[self.elbow_dof_id],
            ],
            dtype=np.float32,
        )
        ee_pos = self._end_effector_pos()
        error = self.target_pos - ee_pos

        return np.array(
            [
                np.cos(q1),
                np.sin(q1),
                np.cos(q2),
                np.sin(q2),
                qvel[0] / 6.0,
                qvel[1] / 6.0,
                ee_pos[0] / self.link_reach,
                ee_pos[1] / self.link_reach,
                self.target_pos[0] / self.link_reach,
                self.target_pos[1] / self.link_reach,
                error[0] / self.link_reach,
                error[1] / self.link_reach,
            ],
            dtype=np.float32,
        )

    def _get_info(self, torque, success: bool):
        ee_pos = self._end_effector_pos()
        ee_vel = self._end_effector_vel()
        error = self.target_pos - ee_pos

        return {
            "ee_x": float(ee_pos[0]),
            "ee_y": float(ee_pos[1]),
            "target_x": float(self.target_pos[0]),
            "target_y": float(self.target_pos[1]),
            "error_x": float(error[0]),
            "error_y": float(error[1]),
            "distance": float(np.linalg.norm(error)),
            "ee_radius": float(np.linalg.norm(ee_pos)),
            "ee_speed": float(np.linalg.norm(ee_vel)),
            "torque_norm": float(np.linalg.norm(torque)),
            "success": bool(success),
        }

    def _sample_target(self):
        return np.array(
            [
                self.np_random.uniform(0.25, 0.68),
                self.np_random.uniform(-0.35, 0.35),
            ],
            dtype=np.float32,
        )

    def _sync_target_site(self):
        self.model.site_pos[self.target_site_id, 0] = float(self.target_pos[0])
        self.model.site_pos[self.target_site_id, 1] = float(self.target_pos[1])
        self.model.site_pos[self.target_site_id, 2] = 0.08

    def _end_effector_pos(self):
        return self.data.site_xpos[self.ee_site_id, :2].astype(np.float32).copy()

    def _end_effector_vel(self):
        site_jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        site_jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, site_jacp, site_jacr, self.ee_site_id)
        vel = site_jacp[:2] @ self.data.qvel
        return vel.astype(np.float32)

    def _distance_to_target(self):
        return float(np.linalg.norm(self.target_pos - self._end_effector_pos()))

    def _sample_initial_qpos(self):
        # 第一阶段先降低探索难度：
        # 从大致朝前的姿态开始，目标也放在前方工作区。
        # 否则全角度随机 + 全圆随机时，策略很容易学成缩在平均位置。
        q1 = self.np_random.uniform(-0.45, 0.45)
        q2 = self.np_random.uniform(-1.20, -0.20)
        return float(q1), float(q2)
