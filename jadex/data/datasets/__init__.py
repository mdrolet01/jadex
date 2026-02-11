from jadex.data.datasets.base_dataset import BaseDataset
from omegaconf import open_dict


def register_datasets():
    if BaseDataset.already_registered():
        return

    from jadex.data.datasets.cifar import CIFARDataset
    from jadex.data.datasets.imagenet import ImageNetDataset
    from jadex.data.datasets.lafan import LAFANDataset
    from jadex.data.datasets.mnist import MNISTDataset

    CIFARDataset.register()
    ImageNetDataset.register()
    MNISTDataset.register()
    LAFANDataset.register()

    # try:
    #     from genmo.data.datasets.hf_dataset import HuggingFaceDataset

    #     HuggingFaceDataset.register()
    # except Exception as e:
    #     print(f"Couldn't load genmo dataset: {e}")


def create_dataset(cfg, mode, ctx):
    """Creates and returns a dataset instance from the given config and mode ('train' or 'test')."""
    assert mode in ("train", "test"), f"Invalid mode '{mode}'. Expected 'train' or 'test'."

    dataset_cls = BaseDataset.registered[cfg.dataset.name]
    dataset = dataset_cls(cfg, mode, ctx)

    with open_dict(cfg):
        cfg.dataset[f"num_{mode}"] = len(dataset)

    return dataset
