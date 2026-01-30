import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from jadex.algorithms.vae.models import create_vae
from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.base.base_state import BaseState, get_mutable
from jadex.base.registrable import register_all
from jadex.data.datasets import create_dataset
from jadex.data.datasets.base_dataset import BaseDataset
from jadex.global_configs.constants import JADEX_CHECKPOINT_DIR
from jadex.networks.variational.constants import LATENT, X

NUM_PLOTS_PER_LABEL = 10
VAE_CHECKPOINT_NAME = ""
VAE_CHECKPOINT_IDX = 0


class GenEval:
    def __init__(self, cfg: DictConfig, model: BaseVAEModel, train_dataset: BaseDataset):
        self.cfg = cfg
        print(cfg.schedulers.beta)
        self.model = model
        self.train_dataset = train_dataset

    @classmethod
    def create(cls):
        cfg = BaseState.load_cfg(JADEX_CHECKPOINT_DIR / VAE_CHECKPOINT_NAME)
        model = create_vae(cfg)
        train_dataset = create_dataset(cfg, "train", ctx=None)
        state: BaseState = model.init(jax.random.PRNGKey(0))
        state = state.load_checkpoint(
            JADEX_CHECKPOINT_DIR / VAE_CHECKPOINT_NAME, checkpoint_idx=VAE_CHECKPOINT_IDX
        )
        gen_eval = cls(cfg, model, train_dataset)
        return gen_eval, state

    def plot_xmat(self, xmat: jnp.ndarray, state: BaseState):
        num_rows, num_cols = xmat.shape[:2]
        fig, axs = plt.subplots(num_cols, num_rows, figsize=(5, 5), dpi=100)
        axs = axs.reshape(num_rows, num_cols)

        dset_name = self.cfg.dataset.name

        if self.cfg.dataset.scaler_mode == "online":
            xmat = self.model.apply_inverse_scaler(xmat, state.scaler_vars, X)
        elif self.cfg.dataset.scaler_mode == "data":
            xmat = self.train_dataset.apply_inverse_scaler(xmat)

        for i in range(num_rows):
            for j in range(num_cols):
                if dset_name in ("MNISTDataset", "MNISTContinuousDataset"):
                    axs[i, j].imshow(xmat[i, j], cmap="gray")
                else:
                    image = np.clip(xmat[i, j], 0, 255).astype(np.uint8)
                    axs[i, j].imshow(image)
                axs[i, j].axis("off")

        fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
        # img = mplfig_to_npimage(fig)
        # plt.close(fig)
        # return img
        plt.show()

    def generate_and_plot(self, state: BaseState):
        z = self.model.latent_dist.sample_from_prior(jax.random.PRNGKey(0), leading_shape=(10, 10))

        p_x_given_z_params, _ = self.model.generative_model.apply(
            state.variables["generative_model"],
            {LATENT: z},
            train=False,
            mutable=get_mutable(state.variables["generative_model"]),
        )

        xmat = self.model.x_dist.get_expected_value(p_x_given_z_params)

        self.plot_xmat(xmat, state)


def main():
    register_all()

    gen_eval, state = GenEval.create()
    gen_eval.generate_and_plot(state)


if __name__ == "__main__":
    main()
