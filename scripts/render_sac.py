from pathlib import Path
import argparse
import sys
import time

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
        description="Render a trained SAC policy.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=TASK_NAMES,
        help="要可视化的任务名称。",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="可选：指定 SAC checkpoint；不填则使用该任务默认 checkpoint。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="环境 reset 随机种子。",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="不加载模型，直接用随机动作展示环境。",
    )
    return parser.parse_args()


def format_info(info):
    fields = []
    for key in (
        "success",
        "distance",
        "block_distance",
        "ee_block_distance",
        "framed",
        "ball_local_x",
        "ball_local_y",
        "error",
        "x",
        "theta_error",
        "ball_x",
        "center_hold_steps",
        "required_center_hold_steps",
    ):
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, float):
            fields.append(f"{key}={value:+.3f}")
        else:
            fields.append(f"{key}={value}")
    return " | ".join(fields)


def main():
    args = parse_args()

    import torch

    from training.sac import load_sac_actor
    from training.sac_configs import SAC_TASK_CONFIGS

    config = SAC_TASK_CONFIGS[args.task]
    model_path = args.model or config.checkpoint_path(ROOT)

    env = config.env_cls(render_mode="human")

    print("task:", args.task)
    print("algo: SAC")
    if args.random:
        actor = None
        device = None
        print("policy: random actions")
    else:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Train it with: python scripts/train_sac.py --task {args.task}, "
                "or add --random to inspect the environment without a model."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor, checkpoint = load_sac_actor(model_path, device=device)

        print("device:", device)
        print("model:", model_path)
        print("hidden_layers:", tuple(checkpoint.get("hidden_layers", (256, 256))))

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode = 1
    step = 0

    try:
        while True:
            if args.random:
                action_np = env.action_space.sample()
            else:
                obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    action = actor.deterministic_action(obs_tensor)
                action_np = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_return += reward
            step += 1

            if step % 50 == 0 or terminated or truncated:
                extra = format_info(info)
                if extra:
                    extra = " | " + extra
                print(
                    f"episode {episode:03d} | step={step:04d} | "
                    f"return={episode_return:9.2f}{extra}"
                )

            time.sleep(env.model.opt.timestep * env.frame_skip)

            if terminated or truncated:
                episode += 1
                step = 0
                episode_return = 0.0
                obs, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
