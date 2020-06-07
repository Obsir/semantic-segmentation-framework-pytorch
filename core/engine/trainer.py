import datetime
import logging
import time
import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
import nni

loss_dict = {
    "diceloss": smp.utils.losses.DiceLoss,
    "crossentropyloss": smp.utils.losses.CrossEntropyLoss,
    "jaccardloss": smp.utils.losses.JaccardLoss,
}
metric_dict = {
    "iou": smp.utils.metrics.IoU,
    "f-score": smp.utils.metrics.Fscore,
    "accuracy": smp.utils.metrics.Accuracy,
    "recall": smp.utils.metrics.Recall,
    "precision": smp.utils.metrics.Precision,
    "dice": smp.utils.metrics.Dice,
}


def do_train(
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
):
    eval_type = 'dice_score'
    start_epoch = arguments["epoch"]
    if start_epoch == max_epoch - 1:
        return
    logger = logging.getLogger("core.trainer")
    logger.info("Start training")
    model.train()
    start_training_time = time.time()

    loss_type = str(cfg.MODEL.LOSS).lower()
    try:
        loss = loss_dict[loss_type]()
    except Exception:
        raise RuntimeError("Loss function is missed!")
    metrics = []
    try:
        if isinstance(cfg.MODEL.METRICS, (tuple, list)):
            for metric_type in cfg.MODEL.METRICS:
                if metric_type == 'dice':
                    eval_type = 'dice_score'
                elif metric_type == 'iou':
                    eval_type = 'iou_score'
                metrics.append(metric_dict[str(metric_type).lower()](threshold=0.5, ignore_channels=cfg.DATASETS.IGNORE_CHANNELS))
    except Exception:
        raise RuntimeError("Metric does not match!")
    # create epoch runners
    # it is a simple loop of iterating over dataset`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    max_score = arguments.get(eval_type, 0)

    for epoch in range(start_epoch, max_epoch):
        train_logs = train_epoch.run(train_data_loader)
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in train_logs.items()]
        meters = '\t'.join(str_logs)
        logger.info(
            '\t'.join(
                [
                    "Train:",
                    "epoch: {epoch}",
                    "{meters}",
                    "lr: {lr:.6f}",
                    "max mem: {memory:.0f}",
                ]
            ).format(
                epoch=epoch,
                meters=meters,
                lr=optimizer.param_groups[0]["lr"],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            ))

        valid_logs = valid_epoch.run(val_data_loader)
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in valid_logs.items()]
        meters = '\t'.join(str_logs)
        logger.info(
            '\t'.join(
                [
                    "Valid:",
                    "epoch: {epoch}",
                    "{meters}",
                    "max mem: {memory:.0f}",
                ]
            ).format(
                epoch=epoch,
                meters=meters,
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            ))

        nni.report_intermediate_result(valid_logs[eval_type])
        logger.debug('Valid accuracy %g', valid_logs[eval_type])
        logger.debug('Pipe send intermediate result done.')
        arguments["epoch"] = epoch
        if epoch % checkpoint_period == 0:
            # checkpointer.save("model_{:03d}".format(epoch), **arguments)
            arguments["is_best"] = False
            checkpointer.save("model_final", **arguments)
        if max_score < valid_logs[eval_type]:
            max_score = valid_logs[eval_type]
            arguments["is_best"] = True
            arguments[eval_type] = max_score
            checkpointer.save("best_model_{:03d}_{:.4}".format(epoch, max_score), **arguments)

    nni.report_final_result(max_score)
    logger.debug('Best result is %g', max_score)
    logger.debug('Send best result done.')
    checkpointer.load_best()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )
