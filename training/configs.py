from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from envs.block_reach_3d_env import BlockReach3DEnv
from envs.cart_pole_balance_env import CartPoleBalanceEnv
from envs.cart_pole_final_env import CartPoleFinalEnv
from envs.cart_pole_recover_env import CartPoleRecoverEnv
from envs.cart_pole_swing_up_env import CartPoleSwingUpEnv
from envs.cart_pole_wide_recover_env import CartPoleWideRecoverEnv
from envs.cart_reach_env import CartReachEnv
from envs.franka_panda_reach_env import FrankaPandaReachEnv
from envs.two_link_arm_gripper_ball_reach_env import TwoLinkArmGripperBallReachEnv
from envs.two_link_arm_random_reach_env import TwoLinkArmRandomReachEnv
from envs.two_link_arm_reach_env import TwoLinkArmReachEnv


@dataclass(frozen=True)
class PPOTrainConfig:
    name: str
    env_cls: Callable
    checkpoint_name: str
    init_checkpoint_name: str | tuple[str, ...] | None = None
    hidden_layers: tuple[int, ...] = (64, 64)
    seed: int = 0
    total_updates: int = 120
    steps_per_update: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    train_epochs: int = 10
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4

    def checkpoint_path(self, root: Path):
        return root / "checkpoints" / self.checkpoint_name

    def init_checkpoint_path(self, root: Path):
        if self.init_checkpoint_name is None:
            return None
        if isinstance(self.init_checkpoint_name, str):
            return root / "checkpoints" / self.init_checkpoint_name

        checkpoint_paths = [
            root / "checkpoints" / checkpoint_name
            for checkpoint_name in self.init_checkpoint_name
        ]
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path.exists():
                return checkpoint_path
        return checkpoint_paths[-1]


TASK_CONFIGS = {
    "cart_reach": PPOTrainConfig(
        name="cart_reach",
        env_cls=CartReachEnv,
        checkpoint_name="ppo_cart_reach_torch.pt",
        total_updates=120,
        entropy_coef=0.01,
    ),
    "block_reach_3d": PPOTrainConfig(
        name="block_reach_3d",
        env_cls=BlockReach3DEnv,
        checkpoint_name="ppo_block_reach_3d_torch.pt",
        total_updates=150,
        entropy_coef=0.01,
    ),
    "franka_panda_reach": PPOTrainConfig(
        name="franka_panda_reach",
        env_cls=FrankaPandaReachEnv,
        checkpoint_name="ppo_franka_panda_reach_torch.pt",
        hidden_layers=(128, 128, 128),
        seed=8,
        total_updates=350,
        steps_per_update=4096,
        minibatch_size=512,
        entropy_coef=0.004,
        learning_rate=2e-4,
    ),
    "two_link_arm_reach": PPOTrainConfig(
        name="two_link_arm_reach",
        env_cls=TwoLinkArmReachEnv,
        checkpoint_name="ppo_two_link_arm_reach_torch.pt",
        init_checkpoint_name="ppo_two_link_arm_reach_torch.pt",
        seed=5,
        total_updates=220,
        entropy_coef=0.003,
    ),
    "two_link_arm_random_reach": PPOTrainConfig(
        name="two_link_arm_random_reach",
        env_cls=TwoLinkArmRandomReachEnv,
        checkpoint_name="ppo_two_link_arm_random_reach_torch.pt",
        init_checkpoint_name=(
            "ppo_two_link_arm_random_reach_torch.pt",
            "ppo_two_link_arm_reach_torch.pt",
        ),
        seed=6,
        total_updates=320,
        steps_per_update=4096,
        minibatch_size=512,
        entropy_coef=0.001,
        learning_rate=2e-4,
    ),
    "two_link_arm_gripper_ball_reach": PPOTrainConfig(
        name="two_link_arm_gripper_ball_reach",
        env_cls=TwoLinkArmGripperBallReachEnv,
        checkpoint_name="ppo_two_link_arm_gripper_ball_reach_torch.pt",
        init_checkpoint_name="ppo_two_link_arm_gripper_ball_reach_torch.pt",
        seed=7,
        total_updates=300,
        steps_per_update=4096,
        minibatch_size=512,
        entropy_coef=0.0015,
        learning_rate=2e-4,
    ),
    "cart_pole_balance": PPOTrainConfig(
        name="cart_pole_balance",
        env_cls=CartPoleBalanceEnv,
        checkpoint_name="ppo_cart_pole_balance_torch.pt",
        total_updates=120,
        entropy_coef=0.001,
    ),
    "cart_pole_recover": PPOTrainConfig(
        name="cart_pole_recover",
        env_cls=CartPoleRecoverEnv,
        checkpoint_name="ppo_cart_pole_recover_torch.pt",
        init_checkpoint_name="ppo_cart_pole_balance_torch.pt",
        seed=1,
        total_updates=180,
        entropy_coef=0.003,
    ),
    "cart_pole_wide_recover": PPOTrainConfig(
        name="cart_pole_wide_recover",
        env_cls=CartPoleWideRecoverEnv,
        checkpoint_name="ppo_cart_pole_wide_recover_torch.pt",
        init_checkpoint_name="ppo_cart_pole_recover_torch.pt",
        seed=2,
        total_updates=220,
        entropy_coef=0.005,
    ),
    "cart_pole_swing_up": PPOTrainConfig(
        name="cart_pole_swing_up",
        env_cls=CartPoleSwingUpEnv,
        checkpoint_name="ppo_cart_pole_swing_up_torch.pt",
        init_checkpoint_name="ppo_cart_pole_wide_recover_torch.pt",
        hidden_layers=(128, 128, 128),
        seed=3,
        total_updates=300,
        entropy_coef=0.008,
    ),
    "cart_pole_final": PPOTrainConfig(
        name="cart_pole_final",
        env_cls=CartPoleFinalEnv,
        checkpoint_name="ppo_cart_pole_final_torch.pt",
        init_checkpoint_name="ppo_cart_pole_swing_up_torch.pt",
        hidden_layers=(128, 128, 128),
        seed=4,
        total_updates=350,
        entropy_coef=0.006,
    ),
}
