import numpy as np

from .cart_pole_balance_env import CartPoleBalanceEnv


class CartPoleRecoverEnv(CartPoleBalanceEnv):
    """
    更难一点的小车杆恢复任务。

    和 CartPoleBalanceEnv 的区别：
        CartPoleBalanceEnv:
            杆子初始时只偏离竖直向上很小角度，主要学“保持平衡”。

        CartPoleRecoverEnv:
            杆子初始时可能偏离竖直向上约 25 度，并带有更大角速度。
            策略需要先把杆子拉回竖直附近，再继续保持平衡。

    这个任务比纯 balance 难，但仍然比从正下方开始的 swing-up 容易很多。
    """

    def __init__(self, render_mode: str | None = None):
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=600,
            init_x_range=0.12,
            init_x_dot_range=0.25,
            init_theta_range=np.deg2rad(25.0),
            init_theta_dot_range=1.20,
            angle_limit=np.deg2rad(75.0),
        )

