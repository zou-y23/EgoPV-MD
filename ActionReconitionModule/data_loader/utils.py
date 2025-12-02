# add all the utils function

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from einops import rearrange


class DataAugmentationMD(object):
    def __init__(self, is_train=False):
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        self.input_size = 224  # args.input_size
        self.is_train = is_train

    def __call__(self, task_dict):
        keys = list(task_dict.keys())
        ijhw = None

        for key in keys:
            if key == "rgb":
                if self.is_train:
                    if ijhw is None:
                        i = int(np.random.uniform(0, 252 - 224))
                        j = int(np.random.uniform(0, 448 - 224))
                        h = w = 224
                        ijhw = (i, j, h, w)
                    else:
                        i, j, h, w = ijhw

                    trans_list = [
                        transforms.ToTensor(),
                        transforms.Resize([252, 448]),  # half_scale
                        # transforms.RandomCrop(224),
                        transforms.Lambda(
                            lambda x: x[:, i : i + h, j : j + w]
                        ),
                        transforms.ColorJitter(
                            brightness=(0.5, 1.5),
                            contrast=(1),
                            saturation=(0.5, 1.5),
                            hue=(-0.1, 0.1),
                        ),
                        transforms.Normalize(self.rgb_mean, self.rgb_std),
                    ]
                else:
                    trans_list = [
                        transforms.ToTensor(),
                        transforms.Resize([252, 448]),  # half_scale
                        transforms.CenterCrop(224),
                        transforms.Normalize(self.rgb_mean, self.rgb_std),
                    ]
                transform = transforms.Compose(trans_list)
                new_value = []
                for frame in task_dict[key]:
                    try:
                        frame = Image.fromarray(frame)
                    except:
                        print(
                            "weird transforms... ",
                            frame,
                            type(frame),
                            frame.shape,
                        )
                        frame = np.zeros((504, 896, 3)).astype(np.uint8)
                        frame = Image.fromarray(frame)
                    new_value.append(transform(frame))
                new_value = torch.stack(new_value)
                T, C, H, W = new_value.shape
                new_value = rearrange(
                    new_value, " t c h w -> t h w c", t=T, c=C, h=H, w=W
                )
                task_dict[key] = new_value
            elif "depth" in key:
                if self.is_train:
                    if ijhw is None:
                        i = int(np.random.uniform(0, 252 - 224))
                        j = int(np.random.uniform(0, 448 - 224))
                        h = w = 224
                        ijhw = (i, j, h, w)
                    else:
                        i, j, h, w = ijhw

                    trans_list = [
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x / 2**9),
                        transforms.Resize([252, 448]),  # half_scale
                        transforms.Lambda(
                            lambda x: x[:, i : i + h, j : j + w]
                        ),
                        # TF.crop(i, j, h, w),
                    ]
                else:
                    trans_list = [
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x / 2**9),
                        transforms.Resize([252, 448]),  # half_scale
                        transforms.CenterCrop(224),
                    ]
                transform = transforms.Compose(trans_list)
                new_value = []
                for frame in task_dict[key]:
                    try:
                        frame = Image.fromarray(frame)
                    except:
                        print(
                            "weird transforms... ",
                            frame,
                            type(frame),
                            frame.shape,
                        )
                        frame = np.zeros((504, 896, 1)).astype(np.uint8)
                        frame = Image.fromarray(frame)
                    new_value.append(transform(frame))
                new_value = torch.stack(new_value)
                T, C, H, W = new_value.shape
                new_value = rearrange(
                    new_value, " t c h w -> t h w c", t=T, c=C, h=H, w=W
                )
                task_dict[key] = new_value
            else:
                # TODO: maybe depth can be further imporved. 
                task_dict[key] = torch.Tensor(task_dict[key])
        return task_dict
