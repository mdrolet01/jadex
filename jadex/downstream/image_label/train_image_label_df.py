from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from chex import PRNGKey
from omegaconf import DictConfig

from jadex.algorithms.vae.models import create_vae
from jadex.algorithms.vae.train_vae import VaeVisionTrainer
from jadex.base.base_state import BaseState
from jadex.base.registrable import register_all
from jadex.downstream.image_label.models import create_image_label_df_model
from jadex.downstream.image_label.models.image_label_df_model import BaseImageLabelDiscreteFlowModel
from jadex.global_configs import jadex_hydra_main
from jadex.global_configs.constants import JADEX_CHECKPOINT_DIR
from jadex.networks.variational.constants import LABEL, X
from jadex.utils import mplfig_to_npimage
from jadex.utils.plotting import use_backend

RESULTS_DIR = Path(__file__).parent / "results"
NUM_PLOTS_PER_LABEL = 10


class ImageLabelDfTrainer(VaeVisionTrainer):
    def __init__(self, cfg, model, train_dataset, test_dataset):
        super().__init__(cfg, model, train_dataset, test_dataset)
        assert cfg.dataset.in_memory, "Only supports cfg.dataset.in_memory!"
        self.train_labels = jnp.array(self.train_dataset.train_labels, jnp.int32)

    @property
    def exclude_metrics(self):
        return []

    @classmethod
    def create_model(cls, cfg):
        vae_model = create_vae(cfg.vae_cfg)
        vae_state: BaseState = vae_model.init(jax.random.PRNGKey(cfg.train.seed))
        vae_state = vae_state.load_checkpoint(
            JADEX_CHECKPOINT_DIR / cfg.model.vae_checkpoint_name,
            checkpoint_idx=cfg.model.vae_checkpoint_idx,
        )
        model = create_image_label_df_model(cfg, vae_model, vae_state)
        return model

    def get_train_batch(self, rng_key: PRNGKey) -> Dict[str, jnp.ndarray]:
        idxs = jax.random.randint(rng_key, (self.cfg.train.batch_size,), 0, len(self.train_data))
        x = self.train_data[idxs]
        label = self.train_labels[idxs]
        x = self._maybe_mnist_binarize(self.cfg, x)
        if self.cfg.dataset.scaler_mode == "data":
            x = self.train_dataset.apply_scaler(x)
        return {X: x, LABEL: label}

    def plot_xmat(self, xmat, state: BaseState):
        num_rows, num_cols = xmat.shape[:2]
        fig, axs = plt.subplots(num_cols, num_rows, figsize=(5, 5), dpi=100)
        axs = axs.reshape(num_rows, num_cols)

        dset_name = self.cfg.vae_cfg.dataset.name

        if self.cfg.dataset.scaler_mode == "online":
            xmat = self.model.apply_inverse_scaler(xmat, state.scaler_vars, X)
        elif self.cfg.dataset.scaler_mode == "data":
            xmat = self.train_dataset.apply_inverse_scaler(xmat)

        if self.cfg.job.get("export_data", False):
            fname_prefix = f"xmat_{state.step:09d}"
            data_dir = RESULTS_DIR / f"{self.cfg.vae_cfg.model.id}_{self.cfg.vae_cfg.dataset.id}"
            data_dir.mkdir(exist_ok=True, parents=True)
            np.savez(data_dir / f"{fname_prefix}.npz", data=np.array(xmat))

        for i in range(num_rows):
            for j in range(num_cols):
                if dset_name in ("MNISTDataset", "MNISTContinuousDataset"):
                    axs[i, j].imshow(xmat[i, j], cmap="gray")
                else:
                    image = np.clip(xmat[i, j], 0, 255).astype(np.uint8)
                    axs[i, j].imshow(image)
                axs[i, j].axis("off")

        fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
        img = mplfig_to_npimage(fig)

        if self.cfg.job.get("export_data", False):
            plt.savefig(data_dir / f"{fname_prefix}.pdf")
            plt.savefig(data_dir / f"{fname_prefix}.png")

        plt.close(fig)
        return img

    def get_metrics(self, prefix, batch, metrics):
        return {}

    def log_expensive(self, prefix, batch, metrics, state=None):
        assert prefix == "train", "Validation not supported!"
        metrics = {}

        generate_from_labels_fn = jax.jit(self.model.generate_from_labels)

        num_classes = self.cfg.vae_cfg.dataset.num_classes
        if num_classes > 10:
            # For ImageNet, select some random classes
            k_vals = [2, 96, 250, 440, 445, 527, 624, 643, 657, 724]
        else:
            k_vals = np.arange(num_classes)

        xmat = np.zeros((len(k_vals), NUM_PLOTS_PER_LABEL, *self.model.x_dist.shape))

        for i, k in enumerate(k_vals):
            xmat[i] = generate_from_labels_fn(
                state, {LABEL: np.full((NUM_PLOTS_PER_LABEL,), k)}, jax.random.PRNGKey(k)
            )

        with use_backend("agg"):
            np_img = self.plot_xmat(xmat, state)

        metrics["label_generations"] = wandb.Image(np_img)

        return metrics

    def run_validation(self, state, get_placeholder=False):
        raise NotImplementedError


@jadex_hydra_main(config_name="image_label_df_config", config_path="./configs")
def main(cfg: DictConfig):
    register_all()
    BaseImageLabelDiscreteFlowModel.registered[cfg.model.name].merge_cfg(cfg)
    ImageLabelDfTrainer.submit(cfg)


if __name__ == "__main__":
    main()
