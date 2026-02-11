### Training

The `train_vae.py` script is configured to train models locally (by default) or on a [SLURM cluster](./configs/cluster/mycluster.yaml) (by passing `cluster=mycluster`). Simply run: 

```
python train_vae.py model=<MODEL_NAME> dataset=<DATASET_NAME>
```

[Model names](configs/model) ("na" suffix means *Non-Autoregressive*) are:
- daps, dapsna
- gumbel, gumbelna
- grmck, grmckna
- vqvae
- fsq
- vae
- ppo

[Dataset names](configs/dataset) are:
- mnist
- cifar
- imagenet256
- lafan (TODO: release)

**NOTE**: Datasets will be automatically downloaded, but ImageNet must be downloaded manually.