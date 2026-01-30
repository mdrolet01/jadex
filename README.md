<p align="center">
  <img width="70%" src="https://github.com/user-attachments/assets/a78cd1f4-b9e8-45a3-882f-e33fbe898569">

</p>

<h2 align="center">JADEX: A library for statistical machine learning, in JAX</h2>

This branch (tag: iclr2026) contains the code for reproducing the results in [Discrete Variational Autoencoding via Policy Search](https://arxiv.org/pdf/2509.24716).


### Features:

- Train discrete VAEs with a **fully jit-compiled**, high performance, JAX pipeline.
- Configure model/architecture hyperparmeters with **hydra**.
- Easily integrate new models/architectures using the **registry**.
- **Discrete flow matching** for downstream analysis of the learned latent space.
- **Wandb** integration to visualize training results in real time.
- Seamless integration with [LocoMujoco](https://github.com/robfiras/loco-mujoco).
- *Coming Soon:* Implementations of popular RL/robotics methods.


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
 - LAFAN

---

### Installation

```
pip install -e .
```


### Training

Please see instructions and [train script](jadex/algorithms/vae/train_vae.py) located in [jadex/algorithms/vae](jadex/algorithms/vae/README.md).


