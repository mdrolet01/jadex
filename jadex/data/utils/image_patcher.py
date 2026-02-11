import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from flax import struct

from jadex.utils import get_closest_square


@struct.dataclass
class Patcher:
    image_height: int
    image_width: int
    image_channels: int
    num_patches: int
    vert_patch_size: int
    horiz_patch_size: int
    num_vert_patches: int
    num_horiz_patches: int
    image_height_pad: int | None = None
    image_width_pad: int | None = None

    @classmethod
    def create(
        cls,
        image_height,
        image_width,
        image_channels,
        desired_num_patches,
        image_height_pad=None,
        image_width_pad=None,
    ):
        num_vert_patches, num_horiz_patches = get_closest_square(desired_num_patches)
        num_patches = num_vert_patches * num_horiz_patches
        assert num_patches == desired_num_patches, "Try a different number of patches"

        if image_height_pad is None:
            assert image_width_pad is None, "Both must be set"
            height = image_height
            width = image_width
            vert_patch_size = cls.round_up_even(np.ceil(height / num_vert_patches))
            horiz_patch_size = cls.round_up_even(np.ceil(width / num_horiz_patches))
        else:
            assert num_vert_patches == num_horiz_patches, "num patches must be a perfect square"
            assert image_width_pad == image_height_pad, "Currently only square padding is supported"
            log_height = np.log2(image_height_pad)
            assert log_height == int(log_height), "Must be a power of 2"
            assert image_height_pad >= image_height
            assert image_width_pad >= image_width
            height = image_height_pad
            width = image_width_pad
            vert_patch_size = height / num_vert_patches
            horiz_patch_size = width / num_horiz_patches
            assert vert_patch_size == int(vert_patch_size) and horiz_patch_size == int(
                horiz_patch_size
            ), "Must be divisible by num_vert_patches and num_horiz_patches"
            assert vert_patch_size == horiz_patch_size, "Currently only square patches are supported"

        assert vert_patch_size > 0
        assert horiz_patch_size > 0
        assert num_vert_patches > 0
        assert num_horiz_patches > 0

        return cls(
            image_height,
            image_width,
            image_channels,
            num_patches,
            vert_patch_size,
            horiz_patch_size,
            num_vert_patches,
            num_horiz_patches,
            image_height_pad,
            image_width_pad,
        )

    @property
    def pad_img_shape(self):
        if self.image_width_pad is not None:
            assert self.image_height_pad is not None, "Both must be set"
            return (self.image_height_pad, self.image_width_pad, self.image_channels)
        else:
            return (
                self.vert_patch_size * self.num_vert_patches,
                self.horiz_patch_size * self.num_horiz_patches,
                self.image_channels,
            )

    @property
    def patch_dim(self):
        return self.vert_patch_size * self.horiz_patch_size * self.image_channels

    @property
    def h_pad(self):
        if self.image_width_pad is not None:
            padding = (self.image_width_pad - self.image_width) // 2
            assert padding >= 0
        else:
            padding = (self.horiz_patch_size * self.num_horiz_patches - self.image_width) // 2
        return padding

    @property
    def v_pad(self):
        if self.image_height_pad is not None:
            padding = (self.image_height_pad - self.image_height) // 2
            assert padding >= 0
        else:
            padding = (self.vert_patch_size * self.num_vert_patches - self.image_height) // 2
        return padding

    @staticmethod
    def round_up_even(x):
        return int(x) if ((int(x) % 2) == 0) else int(x) + 1

    def patchify_pad_flat(self, img, debug=False):
        lead_shape = img.shape[:-3]
        H, W, C = img.shape[-3:]

        assert H == self.image_height
        assert W == self.image_width
        assert C == self.image_channels

        img_pad = jnp.zeros(((*lead_shape,) + self.pad_img_shape))

        if debug:
            # to visualize padding easier
            img_pad = img_pad.at[..., :, :, 0].set(1.0)

        # put the image in the center of the padded image
        img_pad = img_pad.at[
            ...,
            self.v_pad : self.v_pad + self.image_height,
            self.h_pad : self.h_pad + self.image_width,
            :,
        ].set(img)

        if debug:
            plt.imshow(img_pad[0])
            plt.show()

        patches = rearrange(
            img_pad,
            "... (m a) (n b) c -> ... (m n) a b c",
            a=self.vert_patch_size,
            b=self.horiz_patch_size,
            m=self.num_vert_patches,
            n=self.num_horiz_patches,
        )

        if debug:
            fig, axs = plt.subplots(self.num_vert_patches, self.num_horiz_patches, figsize=(5, 5))
            for ax in axs.flat:
                ax.set_xticks([])
                ax.set_yticks([])

            for i in range(self.num_vert_patches):
                for j in range(self.num_horiz_patches):
                    axs[i, j].imshow(
                        patches[0][i * self.num_horiz_patches + j], vmin=patches.min(), vmax=patches.max()
                    )

            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

        flat_pad_patches = rearrange(
            patches,
            "... (m n) a b c -> ... (m n) (a b c)",
            a=self.vert_patch_size,
            b=self.horiz_patch_size,
            m=self.num_vert_patches,
            n=self.num_horiz_patches,
        )

        assert flat_pad_patches.shape == (*lead_shape, self.num_patches, self.patch_dim)

        return flat_pad_patches

    def stitch_flat_pad_patches(self, flat_pad_patches, debug=False):
        lead_shape = flat_pad_patches.shape[:-2]
        assert flat_pad_patches.shape == (*lead_shape, self.num_patches, self.patch_dim)

        patches = rearrange(
            flat_pad_patches,
            "... (m n) (a b c) -> ... (m n) a b c",
            a=self.vert_patch_size,
            b=self.horiz_patch_size,
            m=self.num_vert_patches,
            n=self.num_horiz_patches,
        )

        padded_og_img = rearrange(
            patches,
            "... (m n) a b c -> ... (m a) (n b) c",
            a=self.vert_patch_size,
            b=self.horiz_patch_size,
            m=self.num_vert_patches,
            n=self.num_horiz_patches,
        )

        og_img = padded_og_img[
            ...,
            self.v_pad : self.v_pad + self.image_height,
            self.h_pad : self.h_pad + self.image_width,
            :,
        ]

        if debug:
            plt.imshow(padded_og_img[0])
            plt.show()

            plt.imshow(og_img[0])
            plt.show()

        return og_img

    def get_resnet_decoder_size(self):
        H, W, C = self.pad_img_shape[-3:]
        assert H == W, "Currently only square padding is supported"
        return H

    def get_resnet_decoder_input(self, embeddings: jnp.ndarray):
        MN, D = embeddings.shape[-2:]
        assert MN == self.num_vert_patches * self.num_horiz_patches
        latent_img = rearrange(
            embeddings, "... (m n) d -> ... m n d", m=self.num_vert_patches, n=self.num_horiz_patches
        )
        return latent_img

    def unpad_resnet_decoder_output(self, padded_img):
        img = padded_img[
            ..., self.v_pad : self.v_pad + self.image_height, self.h_pad : self.h_pad + self.image_width, :
        ]
        return img
