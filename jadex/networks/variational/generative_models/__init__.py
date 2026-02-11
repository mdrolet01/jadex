def register_generative_models():
    from .feedforward_gen import VisionFeedForwardGenerativeModel
    from .resnet_gen import VisionResNetGenerativeModel
    from .transformer_gen import TrajTransformerGenerativeModel

    VisionResNetGenerativeModel.register()
    VisionFeedForwardGenerativeModel.register()
    TrajTransformerGenerativeModel.register()
