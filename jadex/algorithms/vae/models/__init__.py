from .base_vae import BaseVAEModel


def register_vaes():
    if BaseVAEModel.already_registered():
        return

    ##### Default Models #####
    from .daps import DAPSModel, DAPSNonAutoregressiveModel
    from .fsq import FSQModel
    from .gumbel import GRMCKModel, GRMCKNonAutoregressiveModel, GumbelModel, GumbelNonAutoregressiveModel
    from .ppo import PPOModel
    from .vae import VAEModel
    from .vqvae import VQVAEModel

    DAPSModel.register()
    DAPSNonAutoregressiveModel.register()
    VQVAEModel.register()
    VAEModel.register()
    GumbelModel.register()
    GumbelNonAutoregressiveModel.register()
    GRMCKNonAutoregressiveModel.register()
    GRMCKModel.register()
    PPOModel.register()
    FSQModel.register()


def create_vae(cfg):
    vae_cls = BaseVAEModel.registered[cfg.model.name]
    return vae_cls.create(cfg)
