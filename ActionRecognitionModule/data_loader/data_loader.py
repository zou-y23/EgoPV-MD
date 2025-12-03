"""Data loader."""
import torch
from torchvision import datasets, transforms
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from .raw_loader import MultiRawDataset
from .utils import DataAugmentationMD
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def custom_collate(data):
    task_list = data[0][0].keys()
    inputs = pd.Series(dtype=object)
    labels =  pd.Series(dtype=object)
    for task in task_list:
        inputs[task] = torch.stack([d[0][task] for d in data])  
    task_list = data[0][1].keys()

    for task in task_list:
        labels[task] = torch.FloatTensor(np.array([d[1][task] for d in data]))
    return inputs, labels


def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """

    assert split in ["train", "val", "test"]
    root = cfg.PATH_TO_DATA_DIR
    label_root = cfg.LABEL_DIR
    tasks = cfg.TASKS  # default ["rgb", "hands-left", "hands-right"]
    max_samples = cfg.DATA.NUM_FRAMES
    benchmark = cfg.DATA.BENCHMARK
    label_file = cfg.DATA.LABEL_FILE
    class_file = cfg.DATA.CLASS_FILE
    time_range = cfg.DATA.TIME_RANGE  # used for anticipation benchmarks
    load_type = cfg.DATA.LOAD_TYPE  #  choice = ['recogtnion', 'anticipation']
    if split in ["train"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
        is_train = True
    elif split in ["val"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
        is_train = False
    elif split in ["test"]:
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
        is_train = False

    subset = split
    transform = DataAugmentationMD(is_train=is_train)
    dataset = MultiRawDataset(
        root=root,
        label_root=label_root,
        tasks=tasks,
        subset=subset,
        load_type=load_type,
        benchmark=benchmark,
        time_range=time_range,
        max_samples=max_samples,
        label_file=label_file,
        class_file=class_file,
        hand_norm=cfg.DATA.HAND_NORM,
        eye_norm=cfg.DATA.EYE_NORM,
        transform=transform,
    )
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=None,
        worker_init_fn=None,
    )

    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)


if __name__ == "__main__":
    from ..models.VideoEncoder.videoencoder.utils.parser import (
        load_config,
        parse_args,
    )
    from ..models.VideoEncoder.videoencoder.datasets import utils as utils

    args = parse_args()
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)
    cfg = load_config(args)
    train_loader = construct_loader(cfg=cfg, split="train")

    for cur_iter, (data, target) in enumerate(train_loader):
        import pdb

        pdb.set_trace()
