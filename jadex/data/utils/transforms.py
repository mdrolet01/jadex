import math
import numbers
import warnings
from collections.abc import Sequence
from typing import Optional, Tuple, Union

import jax
import numpy as np
from chex import PRNGKey
from PIL import Image


def rand_uniform(key, minval=0.0, maxval=1.0) -> float:
    return float(jax.random.uniform(key, (), minval=minval, maxval=maxval).item())


def rand_int(key, minval: int, maxval: int) -> int:
    return int(jax.random.randint(key, (), minval=minval, maxval=maxval).item())


def _setup_size(size, error_msg: str) -> Tuple[int, int]:
    if isinstance(size, numbers.Number):
        return int(size), int(size)
    if isinstance(size, Sequence):
        if len(size) == 1:
            return size[0], size[0]
        if len(size) == 2:
            return size[0], size[1]
    raise ValueError(error_msg)


class Resize:
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        interpolation=Image.BILINEAR,
        max_size: Optional[int] = None,
        antialias: bool = True,
    ):
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_resized_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        orig_h, orig_w = original_size

        if isinstance(self.size, int):
            size = self.size
            if orig_w < orig_h:
                new_w = size
                new_h = int(math.floor(orig_h * size / orig_w))
            else:
                new_h = size
                new_w = int(math.floor(orig_w * size / orig_h))

            if self.max_size and max(new_h, new_w) > self.max_size:
                scale = self.max_size / max(new_h, new_w)
                new_h = int(math.floor(new_h * scale))
                new_w = int(math.floor(new_w * scale))
        else:
            new_h, new_w = self.size if len(self.size) == 2 else (self.size[0], self.size[0])

        return new_h, new_w

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            is_np = True
        elif not isinstance(img, Image.Image):
            raise TypeError("Input should be a numpy array or PIL Image")
        else:
            is_np = False

        new_h, new_w = self._get_resized_size((img.height, img.width))
        resized = img.resize((new_w, new_h), resample=self.interpolation)

        return np.array(resized) if is_np else resized

    def __repr__(self):
        interp = getattr(self.interpolation, "__name__", str(self.interpolation))
        return f"{self.__class__.__name__}(size={self.size}, interpolation={interp}, max_size={self.max_size}, antialias={self.antialias})"


class CenterCrop:
    def __init__(self, size: Union[int, Sequence[int]]):
        self.size = _setup_size(size, "Please provide only two dimensions (h, w) for size.")

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        is_pil = isinstance(img, Image.Image)

        if is_pil:
            img_np = np.array(img)
        elif isinstance(img, np.ndarray):
            img_np = img
        else:
            raise TypeError("Input must be a NumPy array or PIL Image.")

        crop_h, crop_w = self.size
        img_h, img_w = img_np.shape[:2]

        # Padding if needed
        pad_h, pad_w = max(crop_h - img_h, 0), max(crop_w - img_w, 0)
        if pad_h or pad_w:
            pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
            if img_np.ndim == 3:
                pad += ((0, 0),)
            img_np = np.pad(img_np, pad, mode="constant")

        # Cropping
        top = (img_np.shape[0] - crop_h) // 2
        left = (img_np.shape[1] - crop_w) // 2
        cropped = img_np[top : top + crop_h, left : left + crop_w]

        # Return the correct format
        if is_pil:
            return Image.fromarray(cropped)
        return cropped

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be between 0 and 1")
        self.p = p

    def __call__(
        self, img: Union[np.ndarray, Image.Image], rng_key: PRNGKey
    ) -> Union[np.ndarray, Image.Image]:
        if rand_uniform(rng_key) < self.p:
            if isinstance(img, np.ndarray):
                return np.flip(img, axis=1).copy()
            elif isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                raise TypeError("Input should be a numpy array or PIL Image")
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class RandomResizedCrop:
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: int = Image.BILINEAR,
        antialias: bool = True,
    ):
        self.size = _setup_size(size, "size must be an int or a sequence of length 1 or 2")
        if scale[0] > scale[1] or ratio[0] > ratio[1]:
            warnings.warn("Scale and ratio should be in (min, max) format")
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    @staticmethod
    def get_params(
        img: Union[np.ndarray, Image.Image],
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
        rng_key: PRNGKey,
    ) -> Tuple[int, int, int, int]:

        if isinstance(img, np.ndarray):
            height, width = img.shape[:2]
        elif isinstance(img, Image.Image):
            width, height = img.size
        else:
            raise TypeError("img must be a NumPy array or PIL.Image.Image")

        area = height * width
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

        for _ in range(10):
            keys = jax.random.split(rng_key, 5)
            target_area = area * rand_uniform(keys[1], *scale)
            aspect_ratio = math.exp(rand_uniform(keys[2], *log_ratio))
            w, h = int(round(math.sqrt(target_area * aspect_ratio))), int(
                round(math.sqrt(target_area / aspect_ratio))
            )

            if 0 < w <= width and 0 < h <= height:
                i = rand_int(keys[3], 0, height - h + 1)
                j = rand_int(keys[4], 0, width - w + 1)
                return i, j, h, w

        in_ratio = width / height
        if in_ratio < ratio[0]:
            w = width
            h = int(round(w / ratio[0]))
        elif in_ratio > ratio[1]:
            h = height
            w = int(round(h * ratio[1]))
        else:
            h, w = height, width

        i, j = (height - h) // 2, (width - w) // 2
        return i, j, h, w

    def __call__(self, img: Union[np.ndarray, Image.Image], rng_key: PRNGKey) -> np.ndarray:
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError("Input must be a NumPy array or PIL.Image.Image")

        i, j, h, w = self.get_params(img, self.scale, self.ratio, rng_key)
        cropped = img[i : i + h, j : j + w]
        return np.array(Image.fromarray(cropped).resize(self.size[::-1], resample=self.interpolation))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(size={self.size}, "
            f"scale={tuple(round(s, 4) for s in self.scale)}, "
            f"ratio={tuple(round(r, 4) for r in self.ratio)}, "
            f"interpolation={self.interpolation}, antialias={self.antialias})"
        )
