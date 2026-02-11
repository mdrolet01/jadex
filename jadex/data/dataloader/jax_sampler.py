import itertools
from typing import ClassVar, Dict, Iterable, Iterator, Union

import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from flax import struct
from jadex.base.registrable import Registrable


@struct.dataclass
class SampleBufferData:
    array: np.ndarray
    idx_keys: jax.Array
    prev_unused: np.ndarray
    next_unused: np.ndarray
    prev_unused_keys: np.ndarray
    next_unused_keys: np.ndarray
    ptr: int
    rng_key: PRNGKey
    dataset_len: int
    batch_size: int
    num_workers: int
    min_size: int
    unused_size: int


class SampleBuffer:
    def __init__(self, data: SampleBufferData, ctx):
        self.lock = ctx.Lock()
        self.data = data
        self.sent = np.array([])
        self.sent_keys = np.array([])

    @classmethod
    def create_data(cls, dataset_len: int, batch_size: int, num_workers: int, rng_key: PRNGKey):
        min_size = max(dataset_len, 2 * batch_size * num_workers)
        unused_size = max(dataset_len, 2 * batch_size * num_workers)
        array, idx_keys = cls.get_shuffle(dataset_len, batch_size, min_size, rng_key)
        data = SampleBufferData(
            array=array,
            idx_keys=idx_keys,
            prev_unused=np.zeros((unused_size,), dtype=int) - 1,
            next_unused=np.zeros((unused_size,), dtype=int) - 1,
            prev_unused_keys=jnp.zeros((unused_size, 2), dtype=jnp.uint32),
            next_unused_keys=jnp.zeros((unused_size, 2), dtype=jnp.uint32),
            ptr=0,
            rng_key=rng_key,
            dataset_len=dataset_len,
            batch_size=batch_size,
            num_workers=num_workers,
            min_size=min_size,
            unused_size=unused_size,
        )
        return data

    @classmethod
    def create_from_state(cls, data: SampleBufferData, ctx):
        data = data.replace(
            prev_unused=data.next_unused,
            next_unused=np.zeros((data.unused_size,), dtype=int) - 1,
            prev_unused_keys=data.next_unused_keys,
            next_unused_keys=jnp.zeros((data.unused_size, 2), dtype=jnp.uint32),
        )
        return cls(data, ctx)

    @classmethod
    def get_shuffle(cls, dataset_len, batch_size, min_size, rng_key):
        _idx_key, perm_key = jax.random.split(rng_key)
        nums = np.array([], dtype=int)
        while len(nums) < min_size:
            nums = np.append(nums, np.arange(dataset_len))
        sample_idxs = np.array(jax.random.permutation(perm_key, nums), dtype=int)
        max_divisible_size = (len(sample_idxs) // batch_size) * batch_size
        sample_idxs = sample_idxs[:max_divisible_size]
        idx_keys = jax.random.split(_idx_key, len(sample_idxs))
        return sample_idxs, idx_keys

    def synchronize(self):
        """
        NOTE: This *must* be called after every time the dataloader returns a batch!
        This assumes we are using (by default) prefetching, where len(sent) will always be greater than batch size
        """

        cur_unused = self.sent[self.data.batch_size :]
        num_unused = len(cur_unused)
        self.data.next_unused[:num_unused] = cur_unused
        self.sent = cur_unused

        cur_unused_keys = self.sent_keys[self.data.batch_size :]
        next_unused_keys = self.data.next_unused_keys.at[:num_unused].set(cur_unused_keys)
        self.data = self.data.replace(next_unused_keys=next_unused_keys)
        self.sent_keys = cur_unused_keys

        if self.data.ptr >= len(self.data.array) - 1:
            new_rng_key, shuffle_key = jax.random.split(self.data.rng_key)
            new_array, idx_keys = self.get_shuffle(
                dataset_len=self.data.dataset_len,
                batch_size=self.data.batch_size,
                min_size=self.data.min_size,
                rng_key=shuffle_key,
            )
            with self.lock:
                self.data = self.data.replace(array=new_array, idx_keys=idx_keys, ptr=0, rng_key=new_rng_key)

        return self.data

    def get_next_index(self):
        with self.lock:
            if self.data.prev_unused[0] >= 0:
                data_index = self.data.prev_unused[0]
                idx_key = self.data.prev_unused_keys[0]
                prev_unused = np.roll(self.data.prev_unused, -1)
                prev_unused_keys = jnp.roll(self.data.prev_unused_keys, -1, axis=0)
                prev_unused[-1] = -1
                prev_unused_keys = prev_unused_keys.at[-1].multiply(0)
                self.data = self.data.replace(prev_unused=prev_unused, prev_unused_keys=prev_unused_keys)
            else:
                data_index = self.data.array[self.data.ptr]
                idx_key = self.data.idx_keys[self.data.ptr]
                self.data = self.data.replace(ptr=self.data.ptr + 1)

        self.sent = np.append(self.sent, data_index)
        self.sent_keys = jnp.vstack((self.sent_keys, idx_key)) if self.sent_keys.size else idx_key

        return data_index, idx_key


class BaseSampler(Registrable):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements,
    and may provide a :meth:`__len__` method that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`BaseDataloader`, but is expected in any
              calculation involving the length of a :class:`BaseDataloader`.
    """

    registered: ClassVar[Dict[str, "BaseSampler"]] = dict()

    def __iter__(self) -> Iterator:
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]


class BatchSampler(BaseSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, sampler: Union[BaseSampler, Iterable[int]], batch_size: int, drop_last: bool) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        sampler_iter = iter(self.sampler)
        if self.drop_last:
            # Create multiple references to the same iterator
            args = [sampler_iter] * self.batch_size
            for batch_droplast in zip(*args):
                yield [*batch_droplast]
        else:
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class DefaultSampler(BaseSampler):
    def __init__(self, sample_buffer: SampleBuffer):
        self.buffer = sample_buffer

    def __iter__(self):
        while True:
            index, rng_key = self.buffer.get_next_index()
            yield (index, rng_key)

    def __len__(self):
        return len(self.buffer.data.array)


class StatelessSampler(BaseSampler):
    """
    The StatelessSampler is used when we don't want to call SampleBuffer::synchronize.
    NOTE: This can be useful, since calling synchronize inside of jit is not supported.
    However, we cannot resume training exactly where we left off (hence "stateless").
    """

    def __init__(self, seed: int, sample_buffer: SampleBuffer):
        self.length = int(sample_buffer.data.dataset_len)
        self.rng_key = jax.random.PRNGKey(seed)
        self.indices = np.arange(self.length)
        self.np_rng = np.random.default_rng(seed=seed)
        self.np_rng.shuffle(self.indices)
        self.ptr = 0

    def __iter__(self):
        while True:
            key, self.rng_key = jax.random.split(self.rng_key, 2)
            index = self.indices[self.ptr]
            self.ptr += 1
            if self.ptr == self.length:
                self.np_rng.shuffle(self.indices)
                self.ptr = 0
            yield (index, key)

    def __len__(self):
        return self.length


class ValidationSampler:
    def __init__(self, length: int, seed: int = 0):
        self.length = length
        self.seed = seed

    def __iter__(self):
        rng_key = jax.random.PRNGKey(self.seed)
        for idx in range(self.length):
            rng_key, idx_key = jax.random.split(rng_key)
            yield idx, idx_key

    def __len__(self):
        return self.length


def register_samplers():
    if BaseSampler.already_registered():
        return

    DefaultSampler.register()
    StatelessSampler.register()
