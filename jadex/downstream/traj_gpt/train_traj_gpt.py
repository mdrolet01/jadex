from jadex.algorithms.vae.models import create_vae
from jadex.base.base_trainer import BaseTrainer
from jadex.downstream.traj_gpt.models import create_traj_gpt_model
from jadex.global_configs import jadex_hydra_main
from jadex.networks.variational.constants import GOAL, TEXT, X
from omegaconf import DictConfig


class TrajGptTrainer(BaseTrainer):

    @classmethod
    def submit(cls, cfg):
        raise NotImplementedError("Training pipeline not released yet!")

    @staticmethod
    def create_model(cfg):
        raise NotImplementedError
        # vae_model = create_vae()
        # create_traj_gpt_model(vae_model)


@jadex_hydra_main(config_name="traj_gpt_config", config_path="configs")
def main(cfg: DictConfig):
    TrajGptTrainer.submit(cfg)


if __name__ == "__main__":
    main()
