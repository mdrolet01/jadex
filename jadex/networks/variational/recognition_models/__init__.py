def register_recognition_models():
    from .feedforward_rec import VisionFeedForwardRecognitionModel
    from .label_rec import LabelTransformerDiscreteFlowModel
    from .resnet_rec import VisionResNetRecognitionModel
    from .transformer_rec import (
        TrajTransformerRecognitionModel,
        VisionTransformerRecognitionModel,
    )

    VisionTransformerRecognitionModel.register()
    TrajTransformerRecognitionModel.register()

    VisionFeedForwardRecognitionModel.register()
    VisionResNetRecognitionModel.register()
    LabelTransformerDiscreteFlowModel.register()
