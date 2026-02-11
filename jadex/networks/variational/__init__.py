from jadex.networks.variational.variational_network import VariationalNetwork


def register_variational_networks():
    if VariationalNetwork.already_registered():
        return

    from jadex.networks.variational.baseline_models import register_baseline_models
    from jadex.networks.variational.generative_models import register_generative_models
    from jadex.networks.variational.recognition_models import register_recognition_models

    register_generative_models()
    register_recognition_models()
    register_baseline_models()
