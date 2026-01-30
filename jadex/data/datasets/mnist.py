"""
Adapted from PyTorch (BSD 3-Clause License)
"""

import codecs
import os
import os.path
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
from urllib.error import URLError

import jax
import numpy as np
from PIL import Image

from jadex.data.datasets.base_dataset import DSET
from jadex.data.datasets.base_vision_dataset import VisionDataset
from jadex.data.utils.downloading import check_integrity, download_and_extract_archive, flip_byte_order
from jadex.global_configs.constants import CACHE_DIR
from jadex.networks.variational.constants import LABEL, X

PRNGKey = jax.Array


class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            raise NotImplementedError

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file))
            for file in (self.training_file, self.test_file)
        )

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            errors = []
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except URLError as e:
                    errors.append(e)
                    continue
                break
            else:
                s = f"Error downloading {filename}:\n"
                for mirror, err in zip(self.mirrors, errors):
                    s += f"Tried {mirror}, got:\n{str(err)}\n"
                raise RuntimeError(s)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class MNISTDataset(MNIST):
    def __init__(self, cfg, mode, ctx=None):
        assert mode in ("train", "test")
        assert cfg.dataset.height == cfg.dataset.width == 28, "only 28x28 size supported for MNIST"
        assert cfg.dataset.channels == 1, "only 1 channel supported for MNIST"
        assert cfg.dataset.scaler_mode == "data", "MNIST is binary"
        self.cfg = cfg
        data_dir = CACHE_DIR / "mnist"
        data_dir.mkdir(parents=True, exist_ok=True)
        MNIST.__init__(self, root=data_dir, train=bool(mode == "train"), download=True)

    @property
    def dset_type(self) -> DSET:
        return DSET.IMAGE

    # Dataset is binary
    def _normalize(self, x):
        return x

    def _denormalize(self, x):
        return x

    def _standardize(self, x):
        return x

    def _destandardize(self, x):
        return x

    def get_feature_shape(self, feature):
        return {X: tuple(self.cfg.dataset.shape)}[feature]

    def get(self, sampler_data, worker_data=None):
        index, rng_key = sampler_data
        x, target = super().__getitem__(index)
        x = np.asarray(x).reshape(28, 28, 1)
        # convert to binary
        x = (x > 127).astype(np.int32)
        x = self.apply_scaler(x)

        data = {X: x}
        if self.cfg.dataset.get("include_labels", False):
            data[LABEL] = target

        return data


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: np.uint8,
    9: np.int8,
    11: np.int16,
    12: np.int32,
    13: np.float32,
    14: np.float64,
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> np.ndarray:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()

    # parse
    if sys.byteorder == "little" or sys.platform == "aix":
        magic = get_int(data[0:4])
        nd = magic % 256
        ty = magic // 256
    else:
        nd = get_int(data[0:1])
        ty = get_int(data[1:2]) + get_int(data[2:3]) * 256 + get_int(data[3:4]) * 256 * 256

    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    numpy_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    if sys.byteorder == "big" and not sys.platform == "aix":
        for i in range(len(s)):
            s[i] = int.from_bytes(s[i].to_bytes(4, byteorder="little"), byteorder="big", signed=False)

    parsed = np.frombuffer(bytearray(data), dtype=numpy_type, offset=4 * (nd + 1))

    # The MNIST format uses the big endian byte order, while `np.frombuffer` uses whatever the system uses. In case
    # that is little endian and the dtype has more than one byte, we need to flip them.
    if sys.byteorder == "little" and parsed.itemsize > 1:
        parsed = flip_byte_order(parsed)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.reshape(s)


def read_label_file(path: str) -> np.ndarray:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != np.uint8:
        raise TypeError(f"x should be of dtype np.uint8 instead of {x.dtype}")
    if x.ndim != 1:
        raise ValueError(f"x should have 1 dimension instead of {x.ndim}")
    return x.astype(np.int64)


def read_image_file(path: str) -> np.ndarray:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != np.uint8:
        raise TypeError(f"x should be of dtype np.uint8 instead of {x.dtype}")
    if x.ndim != 3:
        raise ValueError(f"x should have 3 dimensions instead of {x.ndim}")
    return x
