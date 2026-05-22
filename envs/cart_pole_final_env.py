import mujoco
import numpy as np

from .cart_pole_balance_env import angle_normalize
from .cart_pole_swing_up_env import CartPoleSwingUpEnv


class CartPoleFinalEnv(CartPoleSwingUpEnv):
    """
    小车杆最终测试任务。

    任务分成两个阶段，但在同一个 reward 里连续表达：
        1. swing-up：从下方把小球甩到上方。
        2. center-hold：让小球逐渐来到世界中心 x=0 附近，并连续停留 2 秒。

    成功条件：
        小球在上方、杆子比较稳定、小球 x 坐标接近 0，
        并且这个状态连续保持 2 秒。
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)

        self.max_episode_steps = 1200
        self.center_hold_steps = 0
        self.center_hold_seconds = 2.0
        self.required_center_hold_steps = int(
            self.center_hold_seconds / (self.model.opt.timestep * self.frame_skip)
        )

        self.ball_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "ball_site",
        )
        if self.ball_site_id < 0:
            raise ValueError("Site 'ball_site' not found in MuJoCo model.")

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.center_hold_steps = 0
        return obs, info

    def _compute_reward(self, x, x_dot, theta_err, theta_dot, force):
        """
        最终任务奖励函数。

        主要奖励项：
            height_reward:
                小球越高越好，用来引导 swing-up。

            upright_reward:
                杆子越接近竖直向上越好。

            center_penalty:
                小球的世界坐标 x 越接近 0 越好。
                注意这里不是小车 x，而是小球 ball_site 的 x。

            hold_bonus:
                小球已经在中心附近稳定时，每一步额外给奖励。
                这会鼓励它停住，而不是经过中心就跑掉。

            success_bonus:
                连续保持中心稳定 2 秒后给大额奖励，并结束 episode。
        """
        theta = angle_normalize(theta_err + np.pi)
        height = 0.5 * (1.0 - np.cos(theta))
        ball_x = float(self.data.site_xpos[self.ball_site_id, 0])

        height_reward = 5.0 * height
        upright_reward = 1.5 * np.cos(theta_err)

        # 小球越高，越强调把小球带到世界中心。
        center_weight = 0.5 + 4.0 * height
        center_penalty = -center_weight * ball_x**2

        cart_position_penalty = -0.15 * x**2
        cart_velocity_penalty = -0.015 * x_dot**2
        pole_velocity_penalty = -0.006 * theta_dot**2
        force_penalty = -0.0004 * force**2

        reward = (
            0.5
            + height_reward
            + upright_reward
            + center_penalty
            + cart_position_penalty
            + cart_velocity_penalty
            + pole_velocity_penalty
            + force_penalty
        )

        centered_now = (
            height > 0.92
            and abs(theta_err) < np.deg2rad(15.0)
            and abs(theta_dot) < 1.5
            and abs(ball_x) < 0.07
            and abs(x_dot) < 0.7
        )

        hold_bonus = 0.0
        if centered_now:
            self.center_hold_steps += 1
            hold_ratio = self.center_hold_steps / self.required_center_hold_steps
            hold_bonus = 4.0 + 6.0 * min(hold_ratio, 1.0)
            reward += hold_bonus
        else:
            self.center_hold_steps = 0

        terminated = False
        boundary_penalty = 0.0
        if abs(x) > self.x_limit:
            boundary_penalty = -60.0
            reward += boundary_penalty
            terminated = True

        truncated = self.step_count >= self.max_episode_steps
        success = self.center_hold_steps >= self.required_center_hold_steps

        success_bonus = 0.0
        if success:
            success_bonus = 200.0
            reward += success_bonus
            terminated = True

        reward_terms = {
            "height_reward": height_reward,
            "upright_reward": upright_reward,
            "center_penalty": center_penalty,
            "hold_bonus": hold_bonus,
            "cart_position_penalty": cart_position_penalty,
            "cart_velocity_penalty": cart_velocity_penalty,
            "pole_velocity_penalty": pole_velocity_penalty,
            "force_penalty": force_penalty,
            "boundary_penalty": boundary_penalty,
            "success_bonus": success_bonus,
        }

        return reward, terminated, truncated, success, reward_terms

    def _get_info(self, force: float, success: bool, reward_terms=None):
        info = super()._get_info(
            force=force,
            success=success,
            reward_terms=reward_terms,
        )
        info["ball_x"] = float(self.data.site_xpos[self.ball_site_id, 0])
        info["center_hold_steps"] = self.center_hold_steps
        info["required_center_hold_steps"] = self.required_center_hold_steps
        return info

