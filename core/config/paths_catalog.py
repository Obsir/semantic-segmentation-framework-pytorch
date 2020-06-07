import os


class DatasetCatalog(object):
    DATASETS = {
        "custom_dataset_train": {
            "images_dir": "/path/to/custom_dataset/train/img",
            "masks_dir": "/path/to/custom_dataset/train/mask",
            "classes": ['background', 'foreground'],
            "split": "train",
        },
        "custom_dataset_val": {
            "images_dir": "/path/to/custom_dataset/val/img",
            "masks_dir": "/path/to/custom_dataset/val/mask",
            "classes": ['background', 'foreground'],
            "split": "val",
        },
        "custom_dataset_test": {
            "images_dir": "/path/to/custom_dataset/test/img",
            "masks_dir": "/path/to/custom_dataset/test/mask",
            "classes": ['background', 'foreground'],
            "split": "test",
        },
    }

    @staticmethod
    def get(name):
        if 'custom_dataset' in name:
            attrs = DatasetCatalog.DATASETS[name]
            return dict(
                factory="CustomDataset",
                args=attrs,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
