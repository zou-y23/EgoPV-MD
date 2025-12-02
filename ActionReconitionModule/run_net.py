"""Wrapper to train and test an action recognition model."""
import imp
from models.VideoEncoder.videoencoder.utils.misc import launch_job
from models.VideoEncoder.videoencoder.utils.parser import load_config, parse_args
from models.VideoEncoder.videoencoder.utils import logging as logging

from train_net import train
from test_net import test
import subprocess
import os
import os.path as osp

logger = logging.get_logger(__name__)


def get_func(cfg):
    train_func = train
    test_func = test
    return train_func, test_func


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    args.num_gpus = 3 
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)
    cfg = load_config(args)
    cfg.NUM_GPUS = 3

    if cfg.DATA.NUM_FRAMES > 16:
        cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE // 2
        cfg.TEST.BATCH_SIZE = cfg.TEST.BATCH_SIZE // 2
        cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / 2
    train, test = get_func(cfg)

    # # Perform training.
    # if cfg.TRAIN.ENABLE:
    #     launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
