from core.data import datasets

from .custom_dataset import custom_dataset_prediction


def predict(**kwargs):
    """evaluate dataset using different methods based on dataset type.
    """
    if isinstance(kwargs['data_loader'].dataset, datasets.CustomDataset):
        return custom_dataset_prediction(**kwargs)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
