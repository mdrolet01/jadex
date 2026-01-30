def register_data_classes():
    from jadex.data.dataloader import register_dataloader_classes
    from jadex.data.datasets import register_datasets

    register_dataloader_classes()
    register_datasets()
