import datetime
import subprocess
from pathlib import Path

import omegaconf
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from jadex.utils.printing import print_blue, print_green, print_yellow

QUIET_HYDRA_HEADER = {
    "defaults": [
        {"override hydra/hydra_logging": "disabled"},
        {"override hydra/job_logging": "disabled"},
    ],
    "hydra": {"output_subdir": None, "run": {"dir": "."}},
}


def merge_sweep_config(cfg: DictConfig):
    wandb.init(job_type="sweep")
    wandb_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in wandb.config.items()])
    print_yellow("#" * 10 + " WANDB SWEEP CONFIG " + "#" * 10)
    print_yellow(OmegaConf.to_yaml(wandb_cfg))
    cfg = OmegaConf.merge(cfg, wandb_cfg)
    with open_dict(cfg):
        cfg.wandb = DictConfig({"project": wandb.run.project})
    return cfg


def create_sweep(cfg: DictConfig):
    # User must specify wandb project for sweeps
    wandb_project = OmegaConf.select(cfg, "wandb.project", throw_on_missing=True)

    wandb_str = f" --project {wandb_project}"
    if OmegaConf.select(cfg, "wandb.entity", default=None, throw_on_missing=False):
        wandb_str += f" --entity {cfg.wandb.entity}"

    base_cfg = omegaconf.OmegaConf.to_container(cfg)
    sweep_cfg = base_cfg.pop("create_sweep")  # remove to prevent recursive loop
    base_cfg.pop("cluster")  # should be specified by run_sweep
    base_cfg.pop("wandb")  # not needed for running sweep

    base_cfg = OmegaConf.merge(QUIET_HYDRA_HEADER, base_cfg)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_dir = Path("wandb_sweeps")
    data_dir = root_dir / f"{wandb_project}_{date}"
    print_green(f"Creating sweep in {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = data_dir / "base_config.yaml"
    with open(base_cfg_path, "w") as f:
        OmegaConf.save(base_cfg, f)

    sweep_cfg["command"].append(f"--config-path={data_dir.relative_to(root_dir.parent)}")
    sweep_cfg["command"].append(f"--config-name=base_config.yaml")

    sweep_cfg_dest_path = data_dir / "sweep_config.yaml"
    with open(sweep_cfg_dest_path, "w") as f:
        OmegaConf.save(sweep_cfg, f)

    command = f"wandb sweep {sweep_cfg_dest_path}{wandb_str}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    print_blue(result.stderr)

    agent_info_path = data_dir / "agent_info.log"
    with open(agent_info_path, "w") as f:
        f.write(result.stderr)
