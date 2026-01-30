from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import wandb
from jadex.utils import mplfig_to_npimage


@contextmanager
def use_backend(backend):
    # Save current backend
    original_backend = plt.get_backend()
    try:
        plt.switch_backend(backend)
        yield
    finally:
        # Restore original backend
        plt.close("all")
        plt.switch_backend(original_backend)


def _get_np_img_for_image(images, cfg, col_factor=1):
    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images * col_factor)))
    num_rows = int(np.ceil(num_images / float(num_cols)))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * col_factor, 5), dpi=100)
    axs = axs.reshape(num_rows, num_cols)

    if cfg.dataset.name != "MNISTDataset":
        images = np.clip(images, 0, 255).astype(np.uint8)

    for i in range(num_rows * num_cols):
        ax = axs[i // num_cols, i % num_cols]
        if i < num_images:
            if cfg.dataset.name in ("MNISTDataset", "MNISTContinuousDataset"):
                ax.imshow(images[i], cmap="gray")
            else:
                ax.imshow(images[i])
        ax.axis("off")

    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
    img = mplfig_to_npimage(fig)
    plt.close(fig)
    return img


def _get_np_img_for_traj(x_hats, xs, cfg, x_recon=None):
    num_images = xs.shape[-1]
    assert x_hats.shape[-1] == num_images
    if x_recon is not None:
        assert x_recon.shape[-1] == num_images

    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / float(num_cols)))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 8), dpi=100, sharex=True, sharey=True)
    axs = axs.reshape(num_rows, num_cols)

    for i in range(num_rows * num_cols):
        ax = axs[i // num_cols, i % num_cols]
        if i < num_images:
            ax.plot(xs[:, i], color="black")
            ax.plot(x_hats[:, i], color="red")
            if x_recon is not None:
                ax.plot(x_recon[:, i], color="blue")
        ax.label_outer()

    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.08, bottom=0.08)
    img = mplfig_to_npimage(fig)
    plt.close(fig)
    return img


def _plot_prediction(x_hats, xs, cfg, prefix="train", plt_kwargs: dict = {}):
    wandb_metrics = {}
    if cfg.dataset.name in ("MNISTDataset", "CIFARDataset", "ImageNetDataset"):
        x_hats = x_hats[:32]
        xs = xs[:32]
        assert len(x_hats) == len(xs)

        fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
        axs[0].imshow(_get_np_img_for_image(x_hats, cfg, col_factor=2))
        axs[0].axis("off")
        axs[1].imshow(_get_np_img_for_image(xs, cfg, col_factor=2))
        axs[1].axis("off")
        fig.tight_layout()
        plt.close(fig)
        np_img = mplfig_to_npimage(fig)
    elif cfg.dataset.name in ("HuggingFaceDataset", "LAFANDataset"):
        max_num_features = 100
        assert x_hats.shape == xs.shape
        x_hats = x_hats[0, :, :max_num_features]
        xs = xs[0, :, :max_num_features]
        if "x_recon" in plt_kwargs.keys():
            x_recon = plt_kwargs["x_recon"][0, :, :max_num_features]
        else:
            x_recon = None
        assert len(x_hats) == len(xs)
        np_img = _get_np_img_for_traj(x_hats, xs, cfg, x_recon)
    else:
        raise ValueError(f"plot for {cfg.dataset.name} not supported")

    wandb_metrics[f"{prefix}_recon"] = wandb.Image(np_img)
    return wandb_metrics


def plot_prediction(x_hats, xs, cfg, prefix="train", plt_kwargs={}):
    with use_backend("agg"):
        return _plot_prediction(x_hats, xs, cfg, prefix, plt_kwargs)
