import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.config import cfg
from core.data import make_data_loader
from core.solver import make_lr_scheduler
from core.solver import make_optimizer
from core.engine.inference import inference
from core.engine.trainer import do_train
from core.modeling.segmentation_model import build_segmentation_model
from core.utils.checkpoint import SegmentationCheckpointer
from core.utils.collect_env import collect_env_info
from core.utils.imports import import_file
from core.utils.logger import setup_logger
from core.utils.miscellaneous import mkdir
import nni


def train(cfg):
    model = build_segmentation_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    optimizer = make_optimizer(cfg, model)
    # scheduler = make_lr_scheduler(cfg, optimizer)
    scheduler = None

    arguments = {}
    arguments["epoch"] = 0

    output_dir = cfg.OUTPUT_DIR

    max_epoch = cfg.SOLVER.MAX_EPOCH

    checkpointer = SegmentationCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk=True
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    train_data_loader = make_data_loader(
        cfg,
        split='train'
    )
    val_data_loader = make_data_loader(
        cfg,
        split='val'
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        train_data_loader,
        val_data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        max_epoch,
    )

    return model


def run_test(cfg, model):
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
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


def main():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation")
    parser.add_argument(
        "--config-file",
        default="./configs/Encoder_UNet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--skip-test",
        default=True,
        help="whether to run testing script with the best model",
        type=bool,
    )

    args = parser.parse_args()
    tuner_params = nni.get_next_parameter()
    tuner_params_list = list()
    for key, value in tuner_params.items():
        tuner_params_list.append(key)
        tuner_params_list.append(value)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(tuner_params_list)

    cfg.update({'OUTPUT_DIR': os.path.join('./training_dir', os.path.basename(args.config_file).split('.yaml')[0], '_'.join([str(i) for i in tuner_params_list]))})
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.GPU_NUM)
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("core", output_dir)
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    logger.debug(tuner_params)

    best_model = train(cfg)

    if not args.skip_test:
        run_test(cfg, best_model)


if __name__ == "__main__":
    main()
