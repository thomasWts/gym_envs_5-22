import numpy as np

from envs.two_link_arm_reach_env import TwoLinkArmReachEnv


class TwoLinkArmRandomReachEnv(TwoLinkArmReachEnv):
    """Harder two-link arm reach task with wider initial poses and targets."""

    def __init__(self, *args, max_episode_steps: int = 350, **kwargs):
        super().__init__(*args, max_episode_steps=max_episode_steps, **kwargs)

    def _sample_initial_qpos(self):
        q1 = self.np_random.uniform(-0.90, 0.90)
        q2 = self.np_random.uniform(-1.80, 0.40)
        return float(q1), float(q2)

    def _sample_target(self):
        radius = self.np_random.uniform(0.22, 0.68)
        angle = self.np_random.uniform(-1.80, 1.80)
        return np.array(
            [radius * np.cos(angle), radius * np.sin(angle)],
            dtype=np.float32,
        )
