import mujoco
import numpy as np

from .cart_pole_balance_env import CartPoleBalanceEnv, angle_normalize


class CartPoleSwingUpEnv(CartPoleBalanceEnv):
    """
    完整小车杆 swing-up 任务。

    初始状态：
        theta = 0 表示杆子向下，小球在下面。
        theta = pi 表示杆子向上，小球在上面。

    任务目标：
        通过左右推小车，把杆子从下方甩到上方，并尽量保持倒立。

    和前几个 recover 任务的关键区别：
        1. 杆子一开始在下方附近，而不是上方附近。
        2. 杆子角度不再因为偏离倒立就终止。
        3. 奖励函数主要根据高度、倒立程度、速度、小车位置和用力大小组成。
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=900,
            init_x_range=0.06,
            init_x_dot_range=0.10,
            init_theta_range=0.18,
            init_theta_dot_range=0.40,
            angle_limit=np.inf,
        )
        self.height_success_steps = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # 覆盖父类 reset：父类会初始化到 theta=pi 附近，这里改成 theta=0 附近。
        self.data.qpos[self.rod_qpos_id] = self.np_random.uniform(
            -self.init_theta_range,
            self.init_theta_range,
        )
        self.data.qvel[self.rod_dof_id] = self.np_random.uniform(
            -self.init_theta_dot_range,
            self.init_theta_dot_range,
        )
        self.height_success_steps = 0
        self.data.ctrl[0] = 0.0

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info(force=0.0, success=False)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def _compute_reward(self, x, x_dot, theta_err, theta_dot, force):
        """
        swing-up 奖励函数。

        奖励思路：
            height_reward:
                根据杆子高度给奖励。下方约为 0，上方约为 1。
                这是 swing-up 最重要的引导项。

            upright_reward:
                杆子越接近竖直向上越高。

            balance_bonus:
                真正到上方附近，并且角速度不大时，额外奖励。
                这会鼓励“甩上去以后停住”，而不是只快速路过上方。

            cart_position_penalty:
                小车不要离中心太远。

            velocity_penalty:
                小车速度和杆子角速度不要太大。

            force_penalty:
                控制力不要长期打满。
        """
        theta = angle_normalize(theta_err + np.pi)
        height_reward = 0.5 * (1.0 - np.cos(theta))
        upright_reward = np.cos(theta_err)

        reward = 0.5
        reward += 5.0 * height_reward
        reward += 1.5 * upright_reward

        near_upright = abs(theta_err) < np.deg2rad(18.0)
        slow_pole = abs(theta_dot) < 2.5
        balance_bonus = 0.0
        if near_upright:
            balance_bonus += 3.0
        if near_upright and slow_pole:
            balance_bonus += 6.0
        reward += balance_bonus

        cart_position_penalty = -0.25 * x**2
        cart_velocity_penalty = -0.015 * x_dot**2
        pole_velocity_penalty = -0.006 * theta_dot**2
        force_penalty = -0.0004 * force**2
        reward += (
            cart_position_penalty
            + cart_velocity_penalty
            + pole_velocity_penalty
            + force_penalty
        )

        terminated = False
        boundary_penalty = 0.0
        if abs(x) > self.x_limit:
            boundary_penalty = -50.0
            reward += boundary_penalty
            terminated = True

        stable_now = (
            abs(theta_err) < np.deg2rad(12.0)
            and abs(theta_dot) < 1.5
            and abs(x) < 0.40
        )
        if stable_now:
            self.height_success_steps += 1
        else:
            self.height_success_steps = 0

        truncated = self.step_count >= self.max_episode_steps
        success = self.height_success_steps >= 50

        success_bonus = 0.0
        if success:
            success_bonus = 100.0
            reward += success_bonus
            terminated = True

        reward_terms = {
            "height_reward": 5.0 * height_reward,
            "upright_reward": 1.5 * upright_reward,
            "balance_bonus": balance_bonus,
            "cart_position_penalty": cart_position_penalty,
            "cart_velocity_penalty": cart_velocity_penalty,
            "pole_velocity_penalty": pole_velocity_penalty,
            "force_penalty": force_penalty,
            "boundary_penalty": boundary_penalty,
            "success_bonus": success_bonus,
        }

        return reward, terminated, truncated, success, reward_terms
