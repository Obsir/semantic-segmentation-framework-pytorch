import bisect
import copy
import logging
from torch.utils.data import DataLoader
import torch.utils.data
from core.utils.imports import import_file
import segmentation_models_pytorch as smp
from . import datasets as D

from .transforms import Transforms
from .collate_batch import BatchCollator


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def build_dataset(dataset_list, transforms, preprocessing, dataset_catalog, split):
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        args["preprocessing"] = preprocessing
        args["transforms"] = transforms
        args['split'] = split
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing and validation, return a list of datasets
    if split == 'test':
        return datasets

    # for training and validation, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_batch_data_sampler(
        sampler, images_per_batch
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    return batch_sampler


def make_data_loader(cfg, split='train'):
    if split == 'train':
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH_TRAIN
        shuffle = True
        dataset_list = cfg.DATASETS.TRAIN
    elif split == 'val':
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH_VAL
        shuffle = False
        dataset_list = cfg.DATASETS.VAL
    elif split == 'test':
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH_TEST
        shuffle = False
        dataset_list = cfg.DATASETS.TEST
    else:
        raise RuntimeError("Split not available: {}".format(split))
    num_workers = cfg.DATALOADER.NUM_WORKERS
    paths_catalog = import_file(
        "core.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog

    transforms = Transforms.build_transforms(cfg, split)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.MODEL.ENCODER, cfg.MODEL.ENCODER_WEIGHTS)
    preprocessing = Transforms.get_preprocessing(preprocessing_fn)
    datasets = build_dataset(dataset_list, transforms, preprocessing, DatasetCatalog, split)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle)
        batch_sampler = make_batch_data_sampler(
            sampler, images_per_batch
        )
        # collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
        )
        data_loaders.append(data_loader)
    if split != 'test':
        # during training and validation, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
