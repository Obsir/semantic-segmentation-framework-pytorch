import logging

from .custom_dataset_prediction import do_custom_dataset_prediction


def custom_dataset_prediction(**kwargs):
    return do_custom_dataset_prediction(
        **kwargs,
    )
