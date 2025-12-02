# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""
import sys

sys.path.append(
    "/home/mahdirad/projects/Ember-HoloAssist/holoAssist-benchmark-dev/"
)
import torch
from torchvision import datasets, transforms
from timesformer.utils.parser import load_config, parse_args

from src.data_loader.tsv_action import MultiTSVDataset


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
    # create data loader
    root = "/mnt/fastdata/processed-data/"
    # tasks = ["rgb", "depth", "hands-left", "hands-right", "head", "eye"]
    tasks = ["rgb", "hands-left", "hands-right"]

    subset = "val"
    target_transform = transforms.Lambda(lambda x: x)
    is_valid_file = lambda x: x.endswith(".png")
    prefixes = {"rgb": ""}
    max_images = None
    time_range = 4
    max_samples = 50
    reference = "hands-left"
    label_file = "/home/wanxin/holoAssist-benchmark-dev/src/data_utils/parse_json/data/data_oct-05_psi_class.json"
    class_file = "/home/wanxin/holoAssist-benchmark-dev/src/data_utils/parse_json/data/labels2idx.json"
    batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    if split in ["train"]:
        subset = "train"
        dataset_train = MultiTSVDataset(
            root,
            tasks,
            time_range,
            max_samples,
            subset,
            transform=None,
            label_file=label_file,
            class_file=class_file,
            target_transform=target_transform,
            prefixes=prefixes,
            max_images=max_images,
            reference=reference,
        )
        loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        )
    elif split in ["val"]:
        dataset_val = MultiTSVDataset(
            root,
            tasks,
            time_range,
            max_samples,
            subset,
            transform=None,
            label_file=label_file,
            class_file=class_file,
            target_transform=target_transform,
            prefixes=prefixes,
            max_images=max_images,
            reference=reference,
        )
        loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        )

    return loader


if __name__ == "__main__":
    args = parse_args()
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)
    cfg = load_config(args)
    train_loader = construct_loader(cfg=cfg, split="train")
    import pdb

    pdb.set_trace()
