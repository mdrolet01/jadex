from omegaconf import DictConfig

from .base_distribution import BaseDistribution


def register_distributions():
    if BaseDistribution.already_registered():
        return

    from .bernoulli import Bernoulli
    from .categorical import Categorical, GRMCKCategorical, GumbelSoftmaxCategorical
    from .diagonal_gaussian import DiagonalGaussian, DiagonalGaussianConstantVariance
    from .uniform import Uniform

    Bernoulli.register()
    DiagonalGaussian.register()
    DiagonalGaussianConstantVariance.register()
    Categorical.register()
    GumbelSoftmaxCategorical.register()
    GRMCKCategorical.register()
    Uniform.register()


def create_distribution(dist_cfg: DictConfig):
    dist_cls = BaseDistribution.registered[dist_cfg.name]
    return dist_cls.create(dist_cfg)
