from .configs import TASK_CONFIGS, PPOTrainConfig

__all__ = [
    "PPOTrainConfig",
    "TASK_CONFIGS",
    "load_actor_critic",
    "load_sac_actor",
    "train_ppo",
    "train_sac",
]


def __getattr__(name):
    if name == "load_actor_critic":
        from .checkpoints import load_actor_critic

        return load_actor_critic
    if name == "train_ppo":
        from .ppo import train_ppo

        return train_ppo
    if name == "load_sac_actor":
        from .sac import load_sac_actor

        return load_sac_actor
    if name == "train_sac":
        from .sac import train_sac

        return train_sac
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
