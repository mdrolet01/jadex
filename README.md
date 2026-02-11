<p align="center">
  <img width="70%" src="https://github.com/user-attachments/assets/fec32696-4b8a-4375-9f90-7605d290b569" />
</p>

<h2 align="center">JADEX: A library for statistical machine learning, in JAX</h2>


This branch contains the code for reproducing the results in [Discrete Variational Autoencoding via Policy Search (ICLR 2026)](https://www.drolet.io/daps/).


### Features:

- Train discrete VAEs with a **fully jit-compiled**, high performance, JAX pipeline.
- Configure model/architecture hyperparameters with **hydra**.
- Easily integrate new models/architectures using the **registry**.
- **Discrete flow matching** for downstream analysis of the learned latent space.
- **Wandb** integration to visualize training results in real time.


### Models:
- DAPS
- VQ-VAE
- FSQ
- Gumbel-Softmax
- GR-MCK
- VAE

### Datasets:
 - ImageNet-1k
 - CIFAR-10
 - MNIST
 - LAFAN (TODO: release)

---

### Installation

```
pip install -e .
```


### Training

Please see instructions and [train script](jadex/algorithms/vae/train_vae.py) located in [jadex/algorithms/vae](jadex/algorithms/vae/README.md).


