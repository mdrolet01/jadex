"""
Adapted from PyTorch (BSD 3-Clause License)
"""

import bisect
from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, Dict, Iterable, Type

import jax.numpy as jnp
from jadex.base.registrable import Registrable
from jadex.data.dataloader.jax_dataloader import stack_collate_fn
from typing_extensions import deprecated


class DSET(Enum):
    IMAGE = "image"
    TRAJECTORY = "trajectory"


class BaseDataset(ABC, Registrable):
    r"""An abstract class representing a :class:`BaseDataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~BaseSampler` implementations and the default options
    of :class:`~BaseDataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~BaseDataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    registered: ClassVar[Dict[str, Type["BaseDataset"]]] = dict()

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of BaseDataset should implement __getitem__.")

    def __add__(self, other: "BaseDataset") -> "ConcatDataset":
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in dataloader/jax_sampler.py

    @classmethod
    def get_dataloader_kwargs(cls, cfg, mode):
        defaults = dict(
            batch_size=int(cfg[mode].batch_size),
            drop_last=True,
            num_workers=int(cfg[mode].num_workers),
            collate_fn=stack_collate_fn,
            prefetch_factor=1,
        )
        return defaults

    @property
    @abstractmethod
    def dset_type(self) -> DSET:
        raise NotImplementedError

    @abstractmethod
    def get_feature_shape(self, feature_alias):
        raise NotImplementedError

    def get_feature_from_batch(self, batch, feature):
        return batch[feature]

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def std(self):
        raise NotImplementedError

    def _normalize(self, x) -> jnp.ndarray:
        raise NotImplementedError

    def _denormalize(self, x) -> jnp.ndarray:
        raise NotImplementedError

    def _standardize(self, x):
        return (x - self.mean) / self.std

    def _destandardize(self, x):
        return x * self.std + self.mean

    @property
    def scaler_type(self):
        return self.cfg.dataset.get("scaler_type", "standardize")

    def apply_scaler(self, x):
        if self.scaler_type == "normalize":
            return self._normalize(x)
        elif self.scaler_type == "standardize":
            return self._standardize(x)
        else:
            raise ValueError

    def apply_inverse_scaler(self, x):
        if self.scaler_type == "normalize":
            return self._denormalize(x)
        elif self.scaler_type == "standardize":
            return self._destandardize(x)
        else:
            raise ValueError

    def compute_l2_loss(self, xs: jnp.ndarray, descaled_x_hats: jnp.ndarray) -> float:
        if self.cfg.dataset.scaler_mode == "data":
            xs = self.apply_inverse_scaler(xs)
            xs = self._standardize(xs)
            x_hats = self._standardize(descaled_x_hats)
        else:
            x_hats = descaled_x_hats

        l2_loss = jnp.sqrt(jnp.square(x_hats - xs).sum(axis=tuple(range(1, xs.ndim)))).mean()
        return l2_loss


class ConcatDataset(BaseDataset):
    r"""BaseDataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: list[BaseDataset]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[BaseDataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    @deprecated("`cummulative_sizes` attribute is renamed to `cumulative_sizes`", category=FutureWarning)
    def cummulative_sizes(self):
        return self.cumulative_sizes
