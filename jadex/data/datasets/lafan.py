from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from omegaconf import open_dict

from jadex.data.datasets.base_dataset import DSET, BaseDataset
from jadex.global_configs.constants import CACHE_DIR
from jadex.networks.variational.constants import X
from jadex.utils.printing import print_green


class LAFANDataset(BaseDataset):
    def __init__(self, cfg, mode, ctx):
        self.cfg = cfg

        lafan_data_dir = CACHE_DIR / "lafan"
        raw_train_data_path = lafan_data_dir / "train_data.npz"
        raw_val_data_path = lafan_data_dir / "val_data.npz"

        # Assumes validation: dance2_subject2, fight1_subject2, run2_subject1, walk2_subject4
        # and train dataset is all other lafan files
        assert raw_train_data_path.exists(), f"{raw_train_data_path} and {raw_val_data_path} do not exist!"
        print_green("Loading cached LAFAN dataset...")
        raw_train_struct = np.load(raw_train_data_path)
        raw_train_data = raw_train_struct["data"]
        train_split_points = raw_train_struct["split_points"]
        raw_val_struct = np.load(raw_val_data_path)
        raw_val_data = raw_val_struct["data"]
        val_split_points = raw_val_struct["split_points"]

        # remove (static) features with variance less than 1e-5
        discard_idxs = np.argwhere(np.var(raw_train_data, axis=0) < 1e-5).flatten()
        discard_vals = raw_train_data[..., discard_idxs].mean(axis=0)
        x_keep_feature_idxs = np.setdiff1d(np.arange(raw_train_data.shape[-1]), discard_idxs)

        train_data = raw_train_data[..., x_keep_feature_idxs]
        val_data = raw_val_data[..., x_keep_feature_idxs]

        with open_dict(cfg):
            cfg.dataset.channels = len(x_keep_feature_idxs)
            cfg.dataset.discard_idxs = discard_idxs.tolist()
            cfg.dataset.discard_vals = np.around(discard_vals, 2).tolist()

        self.x_min = train_data.min(axis=0)
        self.x_max = train_data.max(axis=0)
        self.x_mean = train_data.mean(axis=0)
        self.x_std = train_data.std(axis=0)

        array_fn = jnp.array
        self.train_data = array_fn(train_data)
        self.train_split_points = array_fn(train_split_points)
        self.val_data = array_fn(val_data)
        self.val_split_points = array_fn(val_split_points)

    def __len__(self) -> int:
        # (num files) NOTE: not implemented for this dataset
        return 40

    @property
    def dset_type(self) -> DSET:
        return DSET.TRAJECTORY

    @property
    def mean(self):
        return self.x_mean

    @property
    def std(self):
        return self.x_std

    def _normalize(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)

    def _denormalize(self, x):
        return x * (self.x_max - self.x_min) + self.x_min

    def get_feature_shape(self, feature):
        return {X: tuple(self.cfg.dataset.shape)}[feature]

    def get(self, sampler_data, worker_data=None):
        raise NotImplementedError

    def _get_batch(self, rng_key, mode="train"):
        if mode == "train":
            traj_data = self.train_data
            split_points = self.train_split_points
        elif mode == "test":
            traj_data = self.val_data
            split_points = self.val_split_points
        else:
            raise ValueError

        num_begin_split_indices = len(split_points) - 2
        segment_length = self.cfg.trajectory.input_len

        start_split_idx = jax.random.randint(rng_key, (), 0, num_begin_split_indices)
        start_split = split_points[start_split_idx]
        end_split = split_points[start_split_idx + 1]

        start_idx = jax.random.randint(rng_key, (), start_split, end_split - segment_length - 1)
        x = jax.lax.dynamic_slice_in_dim(traj_data, start_idx, segment_length)

        if self.cfg.dataset.scaler_mode == "data":
            x = self.apply_scaler(x)

        return x

    def get_train_batch(self, rng_key: PRNGKey):
        keys = jax.random.split(rng_key, self.cfg.train.batch_size)
        x = jax.vmap(partial(self._get_batch, mode="train"))(keys)
        return {X: x}

    def get_val_batch(self, rng_key: PRNGKey):
        keys = jax.random.split(rng_key, self.cfg.test.batch_size)
        x = jax.vmap(partial(self._get_batch, mode="test"))(keys)
        return {X: x}
