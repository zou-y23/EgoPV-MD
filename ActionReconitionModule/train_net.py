"""Train a video classification model."""

import numpy as np
import pprint
import torch

from models.VideoEncoder.videoencoder.models import losses as losses
from models.VideoEncoder.videoencoder.models import optimizer as optim
from models.VideoEncoder.videoencoder.utils import checkpoint as cu
from models.VideoEncoder.videoencoder.utils import distributed as du
from models.VideoEncoder.videoencoder.utils import logging as logging
from models.VideoEncoder.videoencoder.utils import metrics as metrics
from models.VideoEncoder.videoencoder.utils import misc as misc
from models.VideoEncoder.videoencoder.visualization import tensorboard_vis as tb
from models.seq2seq.models.build import build_model as build_model_seq
from models.VideoEncoder.videoencoder.models import build_model
from models.VideoEncoder.videoencoder.utils.meters import TrainMeter, ValMeter

# holoassist modifications
from data_loader import data_loader as loader
from einops import rearrange
import wandb
import os.path as osp
from torchmetrics.functional import precision_recall
import mlflow

from models.vit_mdn import vit_base_patch16_224

logger = logging.get_logger(__name__)



def cal_euclidean_dist(output, target):
    point_dist = torch.linalg.norm(output - target, dim=1)
    mean_dist = torch.mean(point_dist)
    return mean_dist


def create_mask(target, hand_xyz, device, clip_dist = None):
    trg_left_hand_avail = target["hands-left"].index_select(
        2, torch.Tensor([0]).to(torch.int64)
    )
    trg_right_hand_avail = target["hands-right"].index_select(
        2, torch.Tensor([0]).to(torch.int64)
    )

    trg_left =  target["hands-left"].index_select(2, hand_xyz).to(device).float()
    trg_right = target["hands-right"].index_select(2, hand_xyz).to(device).float()
    left_mask_temp = (
        trg_left_hand_avail
        .to(torch.bool)
        .squeeze(2)
        .to(device)
    )

    right_mask_temp = (
        trg_right_hand_avail
        .to(torch.bool)
        .squeeze(2)
        .to(device)
    )
    if clip_dist:
    ## when we use clip dist.
        output_left_mask = torch.all(
            torch.logical_and(
                trg_left < clip_dist, trg_left > -clip_dist
            ),
            dim=2,
        ).to(device)
        output_right_mask = torch.all(
            torch.logical_and(
                trg_right < clip_dist, trg_right > -clip_dist
            ),
            dim=2,
        ).to(device)
    else:
        output_left_mask = torch.all(trg_left,dim=2).to(device)
        output_right_mask = torch.all(trg_right,dim=2).to(device)
    left_mask_eval = torch.logical_and(left_mask_temp, output_left_mask)
    right_mask_eval = torch.logical_and(right_mask_temp, output_right_mask)
    return left_mask_eval, right_mask_eval 

def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    #logger.info("device: {}".format(device))
    tasks = cfg.TASKS
    
    if cfg.DATA.BENCHMARK in ["mistake_prediction"]:
        assert False, f"The current version of the code do not support {cfg.DATA.BENCHMARK}."
    
    if cfg.DATA.BENCHMARK == "hand_forecast":
        epoch_dists = [0 for x in range(len(cfg.DATA.EVAL_LEN))]
        hand_xyz = []
        for kk in range(26):
            hand_xyz += [4 + kk * 16, 8 + kk * 16, 12 + kk * 16]
        hand_xyz = torch.Tensor(hand_xyz).to(torch.int64)

    for cur_iter, (data, target) in enumerate(train_loader):
        #Transfer the data to the current GPU device.
        inputs_size = data[tasks[0]][0].size(0)
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MODEL.CLASS_WEIGHTS:
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(
                weight=torch.Tensor(cfg.MODEL.CLASS_WEIGHTS).to(device),
                reduction="mean",
            )
        else:
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        
        # logger.info(preds.shape)
        if cfg.DATA.BENCHMARK == "hand_forecast":

            left_mask_eval, right_mask_eval = create_mask(target, hand_xyz ,device, cfg.DATA.CLIP_DIST)
            
            preds = model(
                data ,target, cfg.TASKS, teacher_forcing_ratio=cfg.TRAIN.TEACHER_RATIO,
                #src_seq, trg_seq, img, depth, teacher_forcing_ratio=cfg.TRAIN.TEACHER_RATIO,
            )
            # Calculate loss for hand forecast
            preds = preds.reshape(
                -1, cfg.DATA.NUM_EVAL_FRAMES, cfg.DATA.DIM_NUM * 2
            )
            trg_left =  target["hands-left"].index_select(2, hand_xyz).to(device).float()
            trg_right = target["hands-right"].index_select(2, hand_xyz).to(device).float()
            preds_left = preds[:, :, : cfg.DATA.DIM_NUM]
            preds_right = preds[:, :, cfg.DATA.DIM_NUM :]
            loss = torch.Tensor([0]).to(device)
            if torch.any(left_mask_eval):
                left_loss = loss_fun(
                    preds_left[left_mask_eval].reshape(-1, 3),
                    trg_left[left_mask_eval].reshape(-1, 3),
                )
                loss += left_loss
            if torch.any(right_mask_eval):
                right_loss = loss_fun(
                    preds_right[right_mask_eval].reshape(-1, 3),
                    trg_right[right_mask_eval].reshape(-1, 3),
                )
                loss += right_loss

            loss /= 2
        else:
            inputs = {}
            for task in tasks:
                if task == "rgb" or "depth" in task:
                    x = data[task].to(device).float()
                    x = rearrange(x, "b t h w c -> b c t h w")
                else:
                    x = data[task].to(device).float()
                inputs[task] = x
            preds = model(**inputs)  # rgb: batch_size, 2283
            # Compute the loss.
            target = target.to(device)
            loss = loss_fun(preds, target)

        try:
            # check Nan Loss.
            misc.check_nan_losses(loss)
        except:
            print(data)
            import pdb

            pdb.set_trace()

        if cur_global_batch_size >= cfg.GLOBAL_BATCH_SIZE:
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            if cfg.MODEL.MODEL_NAME == "seq2seq":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # Update the parameters.
            optimizer.step()
        else:
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward()
            if (cur_iter + 1) % num_iters == 0:
                # for p in model.parameters():
                #     p.grad /= num_iters
                if cfg.MODEL.MODEL_NAME == "seq2seq":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

        # compute errors
        if (cur_iter + 1) % 10 == 0:
            top1_err, top5_err = None, None
            if cfg.DATA.BENCHMARK in [
                "mistake_prediction",
                "intervention_detection",
            ]:
                precisions, recalls = precision_recall(
                    preds,
                    target,
                    num_classes=cfg.MODEL.NUM_CLASSES,
                    average=None,
                )
                logger.info(
                    "precision {}, recall {}".format(precisions.mean(), recalls.mean())
                )
                logger.info(f"preds size: {preds.size(0)}, {preds.size(1)}")
                top1_err = (
                    1 - precisions.mean()
                ) * 100.0  
                top5_err = (
                    1 - recalls.mean()
                ) * 100.0 
            elif cfg.DATA.BENCHMARK in ["hand_forecast"]:
                each_dists = [0 for x in range(len(cfg.DATA.EVAL_LEN))]
                for jj, eval_frame in enumerate(cfg.DATA.EVAL_LEN):
                    left_mask = left_mask_eval[:, :eval_frame]
                    right_mask = right_mask_eval[:, :eval_frame]

                    dist_left = cal_euclidean_dist(
                        trg_left[:, :eval_frame][left_mask].reshape(-1, 3),
                        preds_left[:, :eval_frame][left_mask].reshape(-1, 3),
                    )
                    dist_right = cal_euclidean_dist(
                        trg_right[:, :eval_frame][right_mask].reshape(-1, 3),
                        preds_right[:, :eval_frame][right_mask].reshape(-1, 3),
                    )

                    dist = (dist_left + dist_right) / 2
                    each_dists[jj] = float(dist.cpu().detach())
                    epoch_dists[jj] += each_dists[jj]
                logger.info(
                    "0.5: {}, 1.0, {} 1.5: {}".format(
                        each_dists[0], each_dists[1], each_dists[2]
                    )
                )
                top1_err = torch.Tensor([each_dists[0]]).to(device)
                top5_err = torch.Tensor([each_dists[2]]).to(device)
            else:
                # Compute the errors.
                if cfg.MODEL.NUM_CLASSES <= 10:
                    ks = (1, 1)
                else:
                    ks = (1, 5)
                num_topks_correct = metrics.topks_correct(preds, target, ks)
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

            # Copy the stats from GPU to CPU (sync point).
            loss, top1_err, top5_err = (
                loss.item(),
                top1_err.item(),
                top5_err.item(),
            )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs_size
                * max(
                    cfg.NUM_GPUS, 1
                ), 
            )

            # update azure mlflow
            global_step = data_size * cur_epoch + cur_iter
            mlflow.log_metric("train_loss", loss, step=global_step)
            mlflow.log_metric("lr", lr, step=global_step)
            mlflow.log_metric("train_top1_err", top1_err, step=global_step)
            mlflow.log_metric("train_top5_err", top5_err, step=global_step)

            # write to wandb
            if cfg.USE_WANDB:
                if cfg.NUM_GPUS > 1 and torch.cuda.current_device() > 0:
                    pass
                else:
                    wandb.log(
                        {
                            "train_loss": loss,
                            "lr": lr,
                            "train_top1_err": top1_err,
                            "train_top5_err": top5_err,
                        }
                    )

            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    tasks = cfg.TASKS
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    if cfg.DATA.BENCHMARK == "hand_forecast":
        epoch_dists = [0 for x in range(len(cfg.DATA.EVAL_LEN))]
        hand_xyz = []
        for kk in range(26):
            hand_xyz += [4 + kk * 16, 8 + kk * 16, 12 + kk * 16]
        hand_xyz = torch.Tensor(hand_xyz).to(torch.int64)

    for cur_iter, (data, target) in enumerate(val_loader):
        # Transfer the data to the current GPU device.
        
        inputs_size = data[tasks[0]][0].size(0)

        if cfg.DATA.BENCHMARK == "hand_forecast":

            left_mask_eval, right_mask_eval = create_mask(target, hand_xyz ,device)
            
            preds = model(
                data ,target, cfg.TASKS, teacher_forcing_ratio=0,
            )
            val_meter.data_toc()
            preds = preds.reshape(
                -1, cfg.DATA.NUM_EVAL_FRAMES, cfg.DATA.DIM_NUM * 2
            )
            preds_left = preds[:, :, : cfg.DATA.DIM_NUM]
            preds_right = preds[:, :, cfg.DATA.DIM_NUM :]
            trg_left =  target["hands-left"].index_select(2, hand_xyz).to(device).float()
            trg_right = target["hands-right"].index_select(2, hand_xyz).to(device).float()

        else:
            inputs = {}
            for task in tasks:
                if task == "rgb" or "depth" in task:
                    x = data[task].to(device).float()
                    x = rearrange(x, "b t h w c -> b c t h w")
                else:
                    x = data[task].to(device).float()
                inputs[task] = x
            target = target.to(device)
            val_meter.data_toc()
            preds = model(**inputs)
        if cfg.DATA.BENCHMARK in ["mistake_prediction", "intervention_detection"]:
            # default the number of classes should be 3 and we will ignore the precision and recall of class 0
            precisions, recalls = precision_recall(
                preds, target, num_classes=cfg.MODEL.NUM_CLASSES, average=None
            )
            top1_err = (
                1 - precisions.mean()
            ) * 100.0  
            top5_err = (1 - recalls.mean()) * 100.0  
        elif cfg.DATA.BENCHMARK in ["hand_forecast"]:

            each_dists = [0 for x in range(len(cfg.DATA.EVAL_LEN))]
            for jj, eval_frame in enumerate(cfg.DATA.EVAL_LEN):
                left_mask = left_mask_eval[:, :eval_frame]
                right_mask = right_mask_eval[:, :eval_frame]

                dist_left = cal_euclidean_dist(
                    preds_left[:, :eval_frame][left_mask].reshape(-1, 3),
                    trg_left[:, :eval_frame][left_mask].reshape(-1, 3),
                )
                dist_right = cal_euclidean_dist(
                    preds_right[:, :eval_frame][right_mask].reshape(-1, 3),
                    trg_right[:, :eval_frame][right_mask].reshape(-1, 3),
                )

                dist = (dist_left + dist_right) / 2
                each_dists[jj] = float(dist.cpu().detach())
                epoch_dists[jj] += each_dists[jj]
            logger.info(
                "0.5: {}, 1.0, {} 1.5: {}".format(
                    each_dists[0], each_dists[1], each_dists[2]
                )
            )
            top1_err = torch.Tensor([each_dists[0]]).to(device)
            top5_err = torch.Tensor([each_dists[2]]).to(device)
        else:
            # Compute the errors.
            if cfg.MODEL.NUM_CLASSES <= 10:
                ks = (1, 1)
            else:
                ks = (1, 5)
            num_topks_correct = metrics.topks_correct(preds, target, ks)

            # Combine the errors across the GPUs.
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]

        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err = top1_err.item(), top5_err.item()

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            top1_err,
            top5_err,
            inputs_size
            * max(
                cfg.NUM_GPUS, 1
            ),  
        )

        # write to mlflow
        global_step = len(val_loader) * cur_epoch + cur_iter
        mlflow.log_metric("val_top1_err", top1_err, step=global_step)
        mlflow.log_metric("val_top5_err", top5_err, step=global_step)

        # write to wandb
        if cfg.USE_WANDB:
            if cfg.NUM_GPUS > 1 and torch.cuda.current_device() > 0:
                pass
            else:
                wandb.log(
                    {
                        "val_top1_err": top1_err,
                        "val_top5_err": top5_err,
                    }
                )

        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                global_step=len(val_loader) * cur_epoch + cur_iter,
            )

        val_meter.update_predictions(preds, target)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars({"Val/mAP": val_meter.full_map}, global_step=cur_epoch)
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [label.clone().detach() for label in val_meter.all_labels]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [target.cpu() for target in all_labels]
            writer.plot_eval(preds=all_preds, labels=all_labels, global_step=cur_epoch)

    val_meter.reset()


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # autologging with mlflow
    mlflow.autolog()

    # init wandb
    if cfg.USE_WANDB:
        if cfg.NUM_GPUS > 1 and torch.cuda.current_device() > 0:
            pass
        else:
            wandb.init(
                project="holoassist-benchmark",
                group=cfg.DATA.BENCHMARK,
                job_type="eval" if not cfg.TRAIN.ENABLE else "train",
                tags=cfg.TIMESFORMER.ATTENTION_TYPE,
                name=osp.basename(cfg.OUTPUT_DIR),
            )

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    if cfg.DATA.BENCHMARK == "hand_forecast":
        model = build_model_seq(cfg)
    else:
        model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if not cfg.TRAIN.FINETUNE:
        start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    else:
        start_epoch = 0
        cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer)
        is_checkp_epoch = cu.is_checkpoint_epoch(cfg, cur_epoch, None)
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch, None)
        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
            logger.info(
                "Best val so far. min_top1_err: {}, min_top5_err: {}".format(
                    val_meter.min_top1_err, val_meter.min_top5_err
                )
            )

    if writer is not None:
        writer.close()
