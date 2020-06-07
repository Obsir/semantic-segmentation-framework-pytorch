import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import shutil
from core.config import cfg
from core.data import make_data_loader
from core.engine.inference import inference
from core.modeling.segmentation_model import build_segmentation_model
from core.utils.checkpoint import SegmentationCheckpointer
from core.utils.collect_env import collect_env_info
from core.utils.logger import setup_logger
from core.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Inference")
    parser.add_argument(
        "--config-file",
        default="./configs/Encoder_UNet.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = cfg.OUTPUT_DIR
    logger = setup_logger("core", save_dir)
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_segmentation_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = SegmentationCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    dataset_names = cfg.DATASETS.TEST
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.MODEL.ENCODER + '_' + cfg.MODEL.ARCHITECTURE,
                                         dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    else:
        raise RuntimeError("Output directory is missing!")
    test_data_loaders = make_data_loader(cfg, split='test')
    for output_folder, dataset_name, test_data_loader in zip(output_folders, dataset_names, test_data_loaders):
        inference(
            model,
            test_data_loader,
            dataset_name=dataset_name,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,
        )


if __name__ == "__main__":
    main()
