from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TASK_NAMES = (
    "block_reach_3d",
    "cart_pole_balance",
    "cart_pole_final",
    "cart_pole_recover",
    "cart_pole_swing_up",
    "cart_pole_wide_recover",
    "cart_reach",
    "franka_panda_reach",
    "two_link_arm_gripper_ball_reach",
    "two_link_arm_random_reach",
    "two_link_arm_reach",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MuJoCo Gymnasium task with PyTorch SAC.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=TASK_NAMES,
        help="要训练的任务名称。",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="可选：覆盖默认 SAC 环境步数。",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="可选：从指定 SAC checkpoint 初始化；不填则使用该任务默认的课程学习 checkpoint。",
    )
    parser.add_argument(
        "--no-init-from",
        action="store_true",
        help="禁用默认初始化，完全从随机网络开始训练。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from training.sac import train_sac
    from training.sac_configs import SAC_TASK_CONFIGS

    config = SAC_TASK_CONFIGS[args.task]
    train_sac(
        config=config,
        root=ROOT,
        total_steps=args.total_steps,
        init_from=args.init_from,
        no_init_from=args.no_init_from,
    )


if __name__ == "__main__":
    main()
