import numpy as np

from .cart_pole_balance_env import CartPoleBalanceEnv


class CartPoleWideRecoverEnv(CartPoleBalanceEnv):
    """
    更高难度的小车杆恢复任务。

    相比 CartPoleRecoverEnv：
        1. 初始角度范围从约 ±25 度扩大到约 ±60 度。
        2. 初始角速度更大。
        3. 允许更大的失败角度，让策略有机会从大偏差中救回来。

    这个任务适合作为完整 swing-up 之前的最后一两个过渡难度之一。
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=700,
            init_x_range=0.18,
            init_x_dot_range=0.40,
            init_theta_range=np.deg2rad(60.0),
            init_theta_dot_range=2.00,
            angle_limit=np.deg2rad(110.0),
        )

