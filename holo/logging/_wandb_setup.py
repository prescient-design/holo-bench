from typing import MutableMapping

import holo


def wandb_setup(cfg):
    """
    Runs `wandb.init` and `wandb.login`.
    The values in `cfg` are logged to the wandb run.
    """
    try:
        import wandb
        from omegaconf import OmegaConf
    except ImportError as err:
        raise ImportError(
            "wandb logging requires `wandb` and `hydra-core`. "
            "Install them with `pip install pytorch-holo[benchmark]`."
        ) from err

    if not hasattr(cfg, "wandb_host"):
        cfg["wandb_host"] = "https://api.wandb.ai"

    if not hasattr(cfg, "wandb_mode"):
        cfg["wandb_mode"] = "online"

    if not hasattr(cfg, "project_name"):
        cfg["project_name"] = "holo"

    if not hasattr(cfg, "exp_name"):
        cfg["exp_name"] = "default_group"

    wandb.login(host=cfg.wandb_host)

    wandb.init(
        project=cfg.project_name,
        mode=cfg.wandb_mode,
        group=cfg.exp_name,
    )
    cfg["job_name"] = wandb.run.name
    cfg["__version__"] = holo.__version__
    log_cfg = flatten_config(OmegaConf.to_container(cfg, resolve=True))
    wandb.config.update(log_cfg)


def flatten_config(d: MutableMapping, parent_key: str = "", sep: str = "/"):
    """
    Flatten a nested mapping into a flat dictionary for logging.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
