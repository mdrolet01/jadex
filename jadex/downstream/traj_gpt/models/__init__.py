from .traj_gpt_model import BaseTrajGptModel


def register_traj_gpt_models():
    if BaseTrajGptModel.already_registered():
        return

    from .traj_gpt_model import DAPSTrajGptModel, VQVAETrajGptModel

    DAPSTrajGptModel.register()
    VQVAETrajGptModel.register()


def create_traj_gpt_model(cfg, vae_cfg, vae_model, vae_state):
    traj_gpt_model_cls = BaseTrajGptModel.registered[cfg.model.name]
    return traj_gpt_model_cls.create(cfg, vae_cfg, vae_model, vae_state)
