from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


class TwoLinkArmGripperBallReachEnv(gym.Env):
    """Move an open two-finger gripper so it frames a small ball."""

    metadata = {
        "render_modes": ["human"],
        "render_fps": 100,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | Path | None = None,
        max_episode_steps: int = 300,
        frame_skip: int = 5,
    ):
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        if model_path is None:
            model_path = (
                Path(__file__).resolve().parents[1]
                / "models"
                / "two_link_arm_gripper_ball.xml"
            )

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.torque_limit = 3.0
        self.workspace_scale = 0.8
        self.success_distance = 0.085
        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.previous_distance = 0.0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        self.shoulder_joint_id = self._joint_id("shoulder")
        self.elbow_joint_id = self._joint_id("elbow")
        self.gripper_site_id = self._site_id("gripper_center")
        self.ball_site_id = self._site_id("ball_site")
        self.target_site_id = self._site_id("target_site")

        self.shoulder_qpos_id = self.model.jnt_qposadr[self.shoulder_joint_id]
        self.elbow_qpos_id = self.model.jnt_qposadr[self.elbow_joint_id]
        self.shoulder_dof_id = self.model.jnt_dofadr[self.shoulder_joint_id]
        self.elbow_dof_id = self.model.jnt_dofadr[self.elbow_joint_id]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
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

        q1 = self.np_random.uniform(-0.65, 0.65)
        q2 = self.np_random.uniform(-1.55, 0.05)
        self.data.qpos[self.shoulder_qpos_id] = q1
        self.data.qpos[self.elbow_qpos_id] = q2
        self.data.qvel[self.shoulder_dof_id] = self.np_random.uniform(-0.03, 0.03)
        self.data.qvel[self.elbow_dof_id] = self.np_random.uniform(-0.03, 0.03)
        self.data.ctrl[:] = 0.0

        self.ball_pos = self._sample_ball_pos()
        self.target_pos = self._sample_target_pos(self.ball_pos)
        self._sync_sites()
        self.previous_action[:] = 0.0
        self.step_count = 0

        mujoco.mj_forward(self.model, self.data)
        self.previous_distance = self._distance_to_ball()

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

        gripper_pos = self._gripper_pos()
        gripper_vel = self._gripper_vel()
        ball_local = self._ball_in_gripper_frame()
        error = self.ball_pos - gripper_pos
        distance = float(np.linalg.norm(error))
        progress = self.previous_distance - distance
        action_delta = action - self.previous_action

        centered = abs(ball_local[1]) < 0.050
        inside_opening = abs(ball_local[0]) < 0.080
        framed = centered and inside_opening

        reward = -8.0 * distance
        reward += 20.0 * progress
        reward -= 1.5 * abs(float(ball_local[1]))
        reward -= 0.8 * max(abs(float(ball_local[0])) - 0.08, 0.0)
        reward -= 0.001 * float(np.sum(torque**2))
        reward -= 0.02 * float(np.sum(action_delta**2))

        if framed:
            reward += 3.0
        if distance < 0.16:
            reward -= 0.08 * float(np.linalg.norm(gripper_vel))

        self.previous_distance = distance
        self.previous_action = action.copy()

        success = distance < self.success_distance and framed
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
        gripper_pos = self._gripper_pos()
        error = self.ball_pos - gripper_pos
        ball_local = self._ball_in_gripper_frame()

        return np.array(
            [
                np.cos(q1),
                np.sin(q1),
                np.cos(q2),
                np.sin(q2),
                qvel[0] / 6.0,
                qvel[1] / 6.0,
                gripper_pos[0] / self.workspace_scale,
                gripper_pos[1] / self.workspace_scale,
                self.ball_pos[0] / self.workspace_scale,
                self.ball_pos[1] / self.workspace_scale,
                error[0] / self.workspace_scale,
                error[1] / self.workspace_scale,
                ball_local[0] / 0.18,
                ball_local[1] / 0.12,
            ],
            dtype=np.float32,
        )

    def _get_info(self, torque, success: bool):
        gripper_pos = self._gripper_pos()
        ball_local = self._ball_in_gripper_frame()
        error = self.ball_pos - gripper_pos

        return {
            "gripper_x": float(gripper_pos[0]),
            "gripper_y": float(gripper_pos[1]),
            "ball_x": float(self.ball_pos[0]),
            "ball_y": float(self.ball_pos[1]),
            "target_x": float(self.target_pos[0]),
            "target_y": float(self.target_pos[1]),
            "distance": float(np.linalg.norm(error)),
            "ball_local_x": float(ball_local[0]),
            "ball_local_y": float(ball_local[1]),
            "framed": bool(abs(ball_local[1]) < 0.050 and abs(ball_local[0]) < 0.080),
            "torque_norm": float(np.linalg.norm(torque)),
            "success": bool(success),
        }

    def _sample_ball_pos(self):
        return np.array(
            [
                self.np_random.uniform(0.30, 0.56),
                self.np_random.uniform(-0.22, 0.22),
            ],
            dtype=np.float32,
        )

    def _sample_target_pos(self, ball_pos):
        target = np.array(
            [
                ball_pos[0] + self.np_random.uniform(0.10, 0.22),
                ball_pos[1] + self.np_random.uniform(-0.12, 0.12),
            ],
            dtype=np.float32,
        )
        target[0] = np.clip(target[0], 0.40, 0.72)
        target[1] = np.clip(target[1], -0.35, 0.35)
        return target

    def _sync_sites(self):
        self.model.site_pos[self.ball_site_id, 0] = float(self.ball_pos[0])
        self.model.site_pos[self.ball_site_id, 1] = float(self.ball_pos[1])
        self.model.site_pos[self.ball_site_id, 2] = 0.08
        self.model.site_pos[self.target_site_id, 0] = float(self.target_pos[0])
        self.model.site_pos[self.target_site_id, 1] = float(self.target_pos[1])
        self.model.site_pos[self.target_site_id, 2] = 0.08

    def _joint_id(self, name):
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint {name!r} not found in MuJoCo model.")
        return joint_id

    def _site_id(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Site {name!r} not found in MuJoCo model.")
        return site_id

    def _gripper_pos(self):
        return self.data.site_xpos[self.gripper_site_id, :2].astype(np.float32).copy()

    def _gripper_vel(self):
        site_jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        site_jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            site_jacp,
            site_jacr,
            self.gripper_site_id,
        )
        vel = site_jacp[:2] @ self.data.qvel
        return vel.astype(np.float32)

    def _ball_in_gripper_frame(self):
        gripper_world = self.data.site_xpos[self.gripper_site_id]
        gripper_xmat = self.data.site_xmat[self.gripper_site_id].reshape(3, 3)
        delta_world = np.array(
            [
                self.ball_pos[0] - gripper_world[0],
                self.ball_pos[1] - gripper_world[1],
                0.0,
            ],
            dtype=np.float64,
        )
        return (gripper_xmat.T @ delta_world)[:2].astype(np.float32)

    def _distance_to_ball(self):
        return float(np.linalg.norm(self.ball_pos - self._gripper_pos()))
