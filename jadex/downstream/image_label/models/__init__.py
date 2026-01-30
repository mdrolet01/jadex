from .image_label_df_model import BaseImageLabelDiscreteFlowModel


def register_image_label_df_models():
    if BaseImageLabelDiscreteFlowModel.already_registered():
        return

    from .image_label_df_model import (
        DAPSImageLabelDiscreteFlowModel,
        FSQImageLabelDiscreteFlowModel,
        GRMCKImageLabelDiscreteFlowModel,
        GumbelImageLabelDiscreteFlowModel,
        PPOImageLabelDiscreteFlowModel,
        VQVAEImageLabelDiscreteFlowModel,
    )

    DAPSImageLabelDiscreteFlowModel.register()
    PPOImageLabelDiscreteFlowModel.register()
    GumbelImageLabelDiscreteFlowModel.register()
    GRMCKImageLabelDiscreteFlowModel.register()
    VQVAEImageLabelDiscreteFlowModel.register()
    FSQImageLabelDiscreteFlowModel.register()


def create_image_label_df_model(cfg, vae_model, vae_state):
    image_label_model_cls = BaseImageLabelDiscreteFlowModel.registered[cfg.model.name]
    return image_label_model_cls.create(cfg, vae_model, vae_state)
