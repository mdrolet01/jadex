from jadex.data.dataloader.base_dataloader import BaseDataLoader
from jadex.data.dataloader.jax_dataloader import JaxDataLoader
from jadex.data.dataloader.jax_sampler import BaseSampler, StatelessSampler, ValidationSampler


def register_dataloader_classes():
    from jadex.data.dataloader.jax_sampler import register_samplers

    register_samplers()


def create_sampler(cfg, dataset, sample_buffer) -> BaseSampler:
    sampler_cls = BaseSampler.registered[cfg.dataset.sampler.name]
    return sampler_cls(sample_buffer)


def create_dataloader(cfg, mode, dataset, sample_buffer=None, ctx=None, wrap_jax=True):
    dl_kwargs = dataset.get_dataloader_kwargs(cfg, mode)

    if mode == "train":
        assert sample_buffer is not None
        if cfg.dataset.in_memory:
            dl_kwargs["sampler"] = StatelessSampler(cfg.train.seed, sample_buffer)
        else:
            dl_kwargs["sampler"] = create_sampler(cfg, dataset, sample_buffer)
    else:
        dl_kwargs["sampler"] = ValidationSampler(length=cfg.dataset[f"num_{mode}"])

    dataloader = BaseDataLoader(dataset, **dl_kwargs, multiprocessing_context=ctx)
    if wrap_jax:
        dataloader = JaxDataLoader(dataloader)
    return dataloader
