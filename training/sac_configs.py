from dataclasses import dataclass
from pathlib import Path

from training.configs import TASK_CONFIGS


@dataclass(frozen=True)
class SACTrainConfig:
    name: str
    env_cls: object
    checkpoint_name: str
    init_checkpoint_name: str | tuple[str, ...] | None = None
    seed: int = 0
    total_steps: int = 150_000
    start_steps: int = 5_000
    update_after: int = 2_000
    update_every: int = 50
    batch_size: int = 256
    replay_size: int = 300_000
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    hidden_layers: tuple[int, ...] = (256, 256)
    alpha: float | None = None
    save_every: int = 10_000

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


def _env(name):
    return TASK_CONFIGS[name].env_cls


SAC_TASK_CONFIGS = {
    "cart_reach": SACTrainConfig(
        name="cart_reach",
        env_cls=_env("cart_reach"),
        checkpoint_name="sac_cart_reach_torch.pt",
        seed=10,
        total_steps=80_000,
        start_steps=2_000,
    ),
    "block_reach_3d": SACTrainConfig(
        name="block_reach_3d",
        env_cls=_env("block_reach_3d"),
        checkpoint_name="sac_block_reach_3d_torch.pt",
        seed=11,
        total_steps=150_000,
    ),
    "franka_panda_reach": SACTrainConfig(
        name="franka_panda_reach",
        env_cls=_env("franka_panda_reach"),
        checkpoint_name="sac_franka_panda_reach_torch.pt",
        seed=20,
        total_steps=500_000,
        start_steps=15_000,
        update_after=5_000,
        replay_size=600_000,
        hidden_layers=(256, 256, 256),
        alpha=None,
        save_every=20_000,
    ),
    "two_link_arm_reach": SACTrainConfig(
        name="two_link_arm_reach",
        env_cls=_env("two_link_arm_reach"),
        checkpoint_name="sac_two_link_arm_reach_torch.pt",
        init_checkpoint_name="sac_two_link_arm_reach_torch.pt",
        seed=12,
        total_steps=180_000,
    ),
    "two_link_arm_random_reach": SACTrainConfig(
        name="two_link_arm_random_reach",
        env_cls=_env("two_link_arm_random_reach"),
        checkpoint_name="sac_two_link_arm_random_reach_torch.pt",
        init_checkpoint_name=(
            "sac_two_link_arm_random_reach_torch.pt",
            "sac_two_link_arm_reach_torch.pt",
        ),
        seed=13,
        total_steps=250_000,
        start_steps=8_000,
    ),
    "two_link_arm_gripper_ball_reach": SACTrainConfig(
        name="two_link_arm_gripper_ball_reach",
        env_cls=_env("two_link_arm_gripper_ball_reach"),
        checkpoint_name="sac_two_link_arm_gripper_ball_reach_torch.pt",
        init_checkpoint_name="sac_two_link_arm_gripper_ball_reach_torch.pt",
        seed=14,
        total_steps=300_000,
        start_steps=10_000,
    ),
    "cart_pole_balance": SACTrainConfig(
        name="cart_pole_balance",
        env_cls=_env("cart_pole_balance"),
        checkpoint_name="sac_cart_pole_balance_torch.pt",
        seed=15,
        total_steps=120_000,
    ),
    "cart_pole_recover": SACTrainConfig(
        name="cart_pole_recover",
        env_cls=_env("cart_pole_recover"),
        checkpoint_name="sac_cart_pole_recover_torch.pt",
        init_checkpoint_name="sac_cart_pole_balance_torch.pt",
        seed=16,
        total_steps=180_000,
    ),
    "cart_pole_wide_recover": SACTrainConfig(
        name="cart_pole_wide_recover",
        env_cls=_env("cart_pole_wide_recover"),
        checkpoint_name="sac_cart_pole_wide_recover_torch.pt",
        init_checkpoint_name="sac_cart_pole_recover_torch.pt",
        seed=17,
        total_steps=250_000,
        start_steps=8_000,
    ),
    "cart_pole_swing_up": SACTrainConfig(
        name="cart_pole_swing_up",
        env_cls=_env("cart_pole_swing_up"),
        checkpoint_name="sac_cart_pole_swing_up_torch.pt",
        init_checkpoint_name="sac_cart_pole_wide_recover_torch.pt",
        seed=18,
        total_steps=350_000,
        start_steps=10_000,
        hidden_layers=(256, 256, 256),
    ),
    "cart_pole_final": SACTrainConfig(
        name="cart_pole_final",
        env_cls=_env("cart_pole_final"),
        checkpoint_name="sac_cart_pole_final_torch.pt",
        init_checkpoint_name="sac_cart_pole_swing_up_torch.pt",
        seed=19,
        total_steps=450_000,
        start_steps=10_000,
        hidden_layers=(256, 256, 256),
    ),
}
