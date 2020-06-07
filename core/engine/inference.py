import logging
import time
import os

import torch
from tqdm import tqdm

from core.config import cfg
from core.data.datasets.prediction import predict
from ..utils.timer import Timer, get_time_str


# def compute_on_dataset(model, data_loader, device):
#     model.eval()
#     # evaluate model on test set
#     test_epoch = smp.utils.train.TestEpoch(
#         model=model,
#         metrics=metrics,
#         device=device,
#     )
#     test_logs = test_epoch.run(data_loader)
#     return test_logs


def inference(
        model,
        data_loader,
        dataset_name,
        device="cuda",
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    logger = logging.getLogger("core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    total_timer.tic()
    predict(
        model=model,
        data_loader=data_loader,
        dataset_name=dataset_name,
        device=device,
        output_folder=output_folder,
        logger=logger,
    )
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img)".format(
            total_time_str, total_time / len(dataset)
        )
    )
