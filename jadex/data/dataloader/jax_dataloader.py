import jax
import numpy as np
from flax import jax_utils

from jadex.data.dataloader.base_dataloader import BaseDataLoader


class JaxDataLoader(BaseDataLoader):
    def __init__(self, base_dataloader: BaseDataLoader, cast_numpy=False, prefetch_size=2):
        self.base_dataloader = base_dataloader
        self.cast_numpy = cast_numpy
        self.prefetch_size = prefetch_size
        self._reset_iterator()

    def synchronize(self):
        return self.base_dataloader.sampler.buffer.synchronize()

    @property
    def num_batches(self):
        try:
            # throw error for infinite sampler
            num_batches = len(self.base_dataloader)
        except:
            num_batches = np.inf

        return num_batches

    def _prepare_data_for_pmap(self, xs):
        local_device_count = jax.local_device_count()

        def _prepare(x):
            if self.cast_numpy:
                x = x._numpy()
            return x.reshape((local_device_count, -1) + x.shape[1:])

        return jax.tree_util.tree_map(_prepare, xs)

    def _reset_iterator(self):
        mapped = map(self._prepare_data_for_pmap, self.base_dataloader)
        self.iterator = jax_utils.prefetch_to_device(mapped, size=self.prefetch_size)

    def reset(self):
        """
        Manually reset the underlying iterator. This is intended for dataloaders using finite samplers,
        where the iterator is exhausted after one iteration of the dataset.
        NOTE: You must set `persistent_workers=True` in the dataloader to use this feature.
        """
        self._reset_iterator()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)


def stack_collate_fn(batch):
    return jax.tree.map(lambda *args: np.stack(args), *batch)
