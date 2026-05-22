from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


def angle_normalize(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class CartPoleBalanceEnv(gym.Env):
    """Balance a MuJoCo cart-pole near the upright position."""

    metadata = {
        "render_modes": ["human"],
        "render_fps": 100,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        model_path: str | Path | None = None,
        max_episode_steps: int = 500,
        frame_skip: int = 5,
        init_x_range: float = 0.05,
        init_x_dot_range: float = 0.05,
        init_theta_range: float = 0.06,
        init_theta_dot_range: float = 0.05,
        angle_limit: float = np.deg2rad(60.0),
    ):
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "models" / "cart_pole_balance.xml"

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.force_limit = 30.0
        self.x_limit = 0.85
        self.angle_limit = float(angle_limit)
        self.init_x_range = float(init_x_range)
        self.init_x_dot_range = float(init_x_dot_range)
        self.init_theta_range = float(init_theta_range)
        self.init_theta_dot_range = float(init_theta_dot_range)
        self.step_count = 0

        self.cart_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "cart_slide_x",
        )
        self.rod_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "rod_hinge_y",
        )

        if self.cart_joint_id < 0:
            raise ValueError("Joint 'cart_slide_x' not found in MuJoCo model.")
        if self.rod_joint_id < 0:
            raise ValueError("Joint 'rod_hinge_y' not found in MuJoCo model.")

        self.cart_qpos_id = self.model.jnt_qposadr[self.cart_joint_id]
        self.rod_qpos_id = self.model.jnt_qposadr[self.rod_joint_id]
        self.cart_dof_id = self.model.jnt_dofadr[self.cart_joint_id]
        self.rod_dof_id = self.model.jnt_dofadr[self.rod_joint_id]

        self.observation_space = spaces.Box(
            low=np.array([-2.0, -10.0, -1.0, -1.0, -10.0], dtype=np.float32),
            high=np.array([2.0, 10.0, 1.0, 1.0, 10.0], dtype=np.float32),
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

        self.step_count = 0
        self.data.qpos[self.cart_qpos_id] = self.np_random.uniform(
            -self.init_x_range,
            self.init_x_range,
        )
        self.data.qpos[self.rod_qpos_id] = np.pi + self.np_random.uniform(
            -self.init_theta_range,
            self.init_theta_range,
        )
        self.data.qvel[self.cart_dof_id] = self.np_random.uniform(
            -self.init_x_dot_range,
            self.init_x_dot_range,
        )
        self.data.qvel[self.rod_dof_id] = self.np_random.uniform(
            -self.init_theta_dot_range,
            self.init_theta_dot_range,
        )
        self.data.ctrl[0] = 0.0

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
        theta = float(self.data.qpos[self.rod_qpos_id])
        theta_dot = float(self.data.qvel[self.rod_dof_id])
        theta_err = angle_normalize(theta - np.pi)

        reward, terminated, truncated, success, reward_terms = self._compute_reward(
            x=x,
            x_dot=x_dot,
            theta_err=theta_err,
            theta_dot=theta_dot,
            force=force,
        )

        obs = self._get_obs()
        info = self._get_info(
            force=force,
            success=success,
            reward_terms=reward_terms,
        )

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
        theta = float(self.data.qpos[self.rod_qpos_id])
        theta_dot = float(self.data.qvel[self.rod_dof_id])
        theta_err = angle_normalize(theta - np.pi)

        return np.array(
            [
                x / 0.8,
                x_dot / 3.0,
                np.sin(theta_err),
                np.cos(theta_err),
                theta_dot / 8.0,
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, x, x_dot, theta_err, theta_dot, force):
        """
        小车杆平衡任务的奖励函数。

        任务目标：
            1. 杆子尽量保持竖直向上。
            2. 小车尽量留在轨道中间附近。
            3. 小车和杆子的速度不要太大。
            4. 控制力不要太大。
            5. 如果撑满整个 episode 没倒、没出界，就算成功。

        角度说明：
            theta = pi 表示杆子竖直向上。
            theta_err = theta - pi，并被归一化到 [-pi, pi]。
            theta_err 越接近 0，杆子越接近竖直向上。

        你最常改的几个系数：
            alive_reward:
                每一步都给的基础奖励。它鼓励 episode 活得更久。

            upright_reward:
                杆子越接近竖直向上越大。
                cos(0) = 1，所以完全竖直向上时这一项最大。

            angle_penalty:
                角度误差惩罚。系数越大，越强迫杆子贴近竖直向上。

            cart_position_penalty:
                小车位置惩罚。系数越大，越强迫小车回到中间。

            cart_velocity_penalty / pole_velocity_penalty:
                速度惩罚。系数越大，系统越倾向于慢下来。

            force_penalty:
                控制力惩罚。系数越大，策略越不愿意用大力。

            success_bonus:
                撑满 max_episode_steps 时给的额外奖励。
        """
        alive_reward = 1.0
        upright_reward = 2.0 * np.cos(theta_err)

        angle_penalty = -1.5 * theta_err**2
        cart_position_penalty = -0.20 * x**2
        cart_velocity_penalty = -0.02 * x_dot**2
        pole_velocity_penalty = -0.02 * theta_dot**2
        force_penalty = -0.0005 * force**2

        reward = (
            alive_reward
            + upright_reward
            + angle_penalty
            + cart_position_penalty
            + cart_velocity_penalty
            + pole_velocity_penalty
            + force_penalty
        )

        terminated = False
        boundary_penalty = 0.0
        fall_penalty = 0.0

        # 小车跑太远，直接失败。
        if abs(x) > self.x_limit:
            boundary_penalty = -20.0
            reward += boundary_penalty
            terminated = True

        # 杆子偏离倒立角度太多，直接失败。
        elif abs(theta_err) > self.angle_limit:
            fall_penalty = -10.0
            reward += fall_penalty
            terminated = True

        truncated = self.step_count >= self.max_episode_steps
        success = truncated and not terminated

        success_bonus = 0.0
        if success:
            success_bonus = 50.0
            reward += success_bonus

        reward_terms = {
            "alive_reward": alive_reward,
            "upright_reward": upright_reward,
            "angle_penalty": angle_penalty,
            "cart_position_penalty": cart_position_penalty,
            "cart_velocity_penalty": cart_velocity_penalty,
            "pole_velocity_penalty": pole_velocity_penalty,
            "force_penalty": force_penalty,
            "boundary_penalty": boundary_penalty,
            "fall_penalty": fall_penalty,
            "success_bonus": success_bonus,
        }

        return reward, terminated, truncated, success, reward_terms

    def _get_info(self, force: float, success: bool, reward_terms=None):
        x = float(self.data.qpos[self.cart_qpos_id])
        x_dot = float(self.data.qvel[self.cart_dof_id])
        theta = float(self.data.qpos[self.rod_qpos_id])
        theta_dot = float(self.data.qvel[self.rod_dof_id])
        theta_err = angle_normalize(theta - np.pi)

        return {
            "x": x,
            "x_dot": x_dot,
            "theta": theta,
            "theta_dot": theta_dot,
            "theta_error": theta_err,
            "force": float(force),
            "success": bool(success),
            "reward_terms": reward_terms or {},
        }
