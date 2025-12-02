from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import json
import logging
import numpy as np
import random
import torch
from torchvision.datasets.vision import VisionDataset
from typing import (
    Callable,
    List,
    Optional,
    Dict,
)
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo
from safeIO import TextFile


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_holoassist_data(label_root, subset):
    filename = osp.join(label_root, f"{subset}_0724.txt")
    with open(filename, "r") as fp:
        lines = [line.strip() for line in fp.readlines()]
    return lines


class FileObject(object):
    def __init__(self, root, tasks, v_list) -> None:
        self.root = root
        self.tasks = tasks
        self.v_list = v_list
        self.index_list = OrderedDict()
        self.file_list = OrderedDict()
        self._collect_indices()

    def _collect_indices(self):
        for video_name in self.v_list.keys():
            if video_name not in self.index_list:
                self.index_list[video_name] = OrderedDict()
                self.file_list[video_name] = OrderedDict()

            # always process rgb-info first
            filename = os.path.join(
                self.root, video_name, "Export_py", "Video", "Pose_sync"
            )
            with open(filename + ".lineidx", "r") as file:
                lines = [int(line.strip()) for line in file.readlines()]
            self.index_list[video_name]["pose"] = lines
            self.file_list[video_name]["pose"] = filename + ".txt"

            for task in self.tasks:
                if "rgb" in task:
                    continue
                if "hands" in task:
                    if "left" in task:
                        filename = os.path.join(
                            self.root,
                            video_name,
                            "Export_py",
                            "Hands",
                            "Left_sync",
                        )
                    else:
                        filename = os.path.join(
                            self.root,
                            video_name,
                            "Export_py",
                            "Hands",
                            "Right_sync",
                        )
                    with open(filename + ".lineidx", "r") as file:
                        lines = [int(line.strip()) for line in file.readlines()]
                    self.index_list[video_name][task] = lines
                    self.file_list[video_name][task] = filename + ".txt"
                else:
                    if "eye" in task:
                        prefix = "Eyes"
                    elif "head" in task:
                        prefix = "Head"
                    else:
                        raise NotImplementedError
                    filename = osp.join(
                        self.root,
                        video_name,
                        "Export_py",
                        prefix,
                        f"{prefix}_sync",
                    )
                    with open(filename + ".lineidx", "r") as file:
                        lines = [int(line.strip()) for line in file.readlines()]
                    self.index_list[video_name][task] = lines
                    self.file_list[video_name][task] = filename + ".txt"
        logger.info("finish loadding the indices")

    def seek_poses(self, poses, f):
        lines = []
        for pos in poses:
            line = f.readline(pos)
            lines.append(line.strip())
        return lines


class MultiRawDataset(VisionDataset):
    """A generic multi-task dataset loader where the samples are arranged in this way: ::
        root/R014-7July-DSLR/Export_py/Video_compress.mp4 (including audio)
        root/R014-7July-DSLR/Export_py/Eyes
        root/R014-7July-DSLR/Export_py/Hands
        root/R014-7July-DSLR/Export_py/Head
        root/R014-7July-DSLR/Export_py/IMU
        root/R014-7July-DSLR/Export_py/AhatDepth_synced.tar
    Args:
        root (string): Root directory path to data files.
        label_root (string): Root directory path to label files and data splits.
        tasks (list): List of tasks as strings.
        subset (str): Choices are ['train', 'val', 'test'] splits.
        load_type (str): Choices are ['recognition', 'anticipation'].
            Indicating different samplings of the video clips.
            Use `anticipation` only for `fine-grained-action anticipation`
            and `coarse-grained-action anticipation` tasks.
        benchmark (str): Benchmark choices used in HoloAssist paper.
            Choices are [fine_grained_action, coarse_grained_action,
            mistake_prediction, intervention_prediction_{1,3,5},
            intervention_detection_{1,3,5}]
        time_range (float): anticipation time window.  Default is 1 second.
            Used only when `load_type` is `anticipation`.
        max_samples (int): number of samples per clip. The default video fps
            is 30fps and we sample `max_samples` frames from any
            given time period.
        label_file (str): `labels_{date}_{seq_num}_classes.json` file
            is a dictinary that shows the clips and its labels in the
            format of (task_id, label_id) for each benchmark
            (dictionary key).
        class_file (str): `labels_{date}_{seq_num}_label2idx.json` file is a
            dictionary that shows the matching between `class_name` and
            `class_index`. The dictionary key is the benchmark name
            (or benchmark prefix for `intervention_prediction` and
            `intervention_detection` which supports 3 different input time
            window {1, 3, 5} seconds).
        hand_norm (bool):  Transform hands compared to the synchronized
            head pose from Video data.  Default is True.
        eye_norm (bool):  Transform hands compared to the synchronized
            head pose from Video data.  Default is True.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    """

    def __init__(
        self,
        root: str,
        label_root: str,
        tasks: List[str],
        subset: str,
        load_type: str,  # choice = ['recogtnion', 'anticipation'] # "anticipatio is always from the beginning of the video until time range (in seconds) as the observation period"
        benchmark: str,  # [fine_grained_action, coarse_grained_action, intervention_prediction_{1,3,5}, intervention_detection_{1,3,5}]
        time_range: float = 1,
        max_samples: int = 16,
        label_file: str = "labels_20230724_2221_classes.json",
        class_file: str = "labels_20230724_2221_label2idx.json",
        hand_norm: bool = True,
        eye_norm: bool = True,
        transform: Callable = None,
        target_transform: Optional[Callable] = None,
    ):
        super(MultiRawDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.transform = transform
        self.target_transform = target_transform
        self._chunk_sizes = None
        self.load_type = load_type
        self.benchmark = benchmark  # return particular labels
        self.root = root
        self.label_root = label_root
        self.subset = subset  # get train, val, test subset
        self.is_train = subset == "train"
        self.tasks = tasks
        self.time_range = time_range
        self.max_samples = max_samples
        self.hand_norm = hand_norm
        self.eye_norm = eye_norm
        self.fps = 30
        self.class_file = class_file = osp.join(self.label_root, class_file)
        self.label_file = label_file = osp.join(self.label_root, label_file)

        # checks to make sure `benchmark`` and `load_type` are compatible.
        if self.benchmark == "mistake_prediction":  # follow Assembly101 definition
            assert (
                load_type == "recognition"
            )  # the label file has been preprocessed so that the start and end time have been changed to coarse_grained action start time to fine grained action end time.
        if "intervention_prediction" in self.benchmark:
            assert load_type == "recognition"

        if "intervention_detection" in self.benchmark:
            assert load_type == "recognition"

        logger.info(
            "load_type: {},  benchmark: {}, time_range: {}, num_frames: {}".format(
                load_type, benchmark, time_range, max_samples
            )
        )

        # load events (clips+lables) and label2idx files.

        # both recognition and anticipation uses the same set of clips
        if "coarse_grained" in benchmark:
            event_key = "coarse_grained_action"
        elif "fine_grained" in benchmark:
            event_key = "fine_grained_action"
        else:
            event_key = benchmark

        logger.info(f"event key is {event_key}")
        self.events = self._load_events(label_file)[
            event_key
        ]  # {vname: list of clips and class ids}
        self.idx2label = self._load_map(
            class_file
        )  #  OrderedDict({"id": "label name"})
        self.num_classes = len(
            self.idx2label
        )  # this should be matching the number of classes in the model

        self.axis_transform = np.linalg.inv(
            np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        )

        # load videos
        self.sessions = sessions = build_holoassist_data(label_root, subset)
        self.v_list = OrderedDict()
        self.v_time_list = OrderedDict()
        for video_name in tqdm(sessions):
            self.v_list[video_name] = os.path.join(
                root, video_name, "Export_py", "Video_compress.mp4"
            )
            with open(
                os.path.join(root, video_name, "Export_py/Video", "VideoMp4Timing.txt"),
                "r",
            ) as fp:  # update data path
                self.v_time_list[video_name] = [x.strip() for x in fp.readlines()]

        self.clip_list = []
        self.file_obj = FileObject(root, tasks, self.v_list)
        self.index_list = self.file_obj.index_list
        self.file_list = self.file_obj.file_list
        self._label_placeholder = 0
        self._collect_clips()

    def _collect_clips(self):
        """regenerate the ids"""
        start_time = datetime.now()
        logger.info("=> start to collect video clips ...")

        for video_name in self.v_list.keys():
            # get action clip
            if not video_name in self.events:
                logger.info(f"ignoring {video_name} as no labels in that video.")
            else:
                events_i = self._get_events(video_name)
                if len(events_i) == 0:
                    logger.info(f"ignoring {video_name} as video is empty.")
                    pass
                else:
                    for idx_event in range(len(events_i)):
                        t_start = events_i[idx_event][0]
                        t_end = events_i[idx_event][1]
                        if t_end < t_start:
                            logger.info(
                                "filtering out invalid clips where t_end < t_start."
                            )
                            continue
                        if int((t_end - t_start) * 30) == 0:
                            logger.info("removing action clips that are too short")
                            continue
                        label = int(events_i[idx_event][2][1])  # [task_id, label_id]
                        # add video name and start_id and end_id in the sequence.
                        if self.load_type == "anticipation":  # with time range
                            observe_end = max(t_start - self.time_range, 0)
                            if observe_end * self.fps > self.max_samples:
                                self.clip_list.append(
                                    [0, observe_end, label, video_name]
                                )
                        elif self.load_type == "recognition":
                            self.clip_list.append([t_start, t_end, label, video_name])
                        else:
                            raise NotImplementedError
                        if idx_event == len(events_i):
                            break

        duration = datetime.now() - start_time

        logger.info(
            "=> finish collecting clips. Takes {}, {} clips".format(
                str(duration), len(self.clip_list)
            )
        )

    def _load_events(self, map_file: str):
        if not map_file:
            return None

        f = open(map_file)
        data = json.load(f)
        f.close()
        return data

    def _get_events(self, video_name):
        return self.events[video_name]

    def _load_map(self, map_file: str):
        if not map_file:
            return None
        with open(map_file, "r") as fp:
            idx2label_all = json.load(fp)
        benchmark = self.benchmark
        if "intervention_prediction" in benchmark:
            benchmark = "intervention_prediction"
        elif "intervention_detection" in benchmark:
            benchmark = "intervention_detection"

        idx2label_list = idx2label_all[benchmark]
        idx2label = OrderedDict()
        for i, label in enumerate(idx2label_list):
            idx2label[str(i)] = label
        return idx2label

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        label = self.clip_list[index][2]
        t_start = self.clip_list[index][0]
        t_end = self.clip_list[index][1]
        video_name = self.clip_list[index][3]

        sample_dict = self._load_sample_dict(video_name, t_start, t_end)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return sample_dict, label

    def _load_sample_dict(self, video_name, t_start, t_end):
        """Loading multimodal samples for each action clip"""
        sample_dict = {}

        # Need to process rgb-info first to get intrinsics and cam pose for
        # norm hand poses
        def _frame_filter(idx_list):
            if self.max_samples > len(idx_list):
                return idx_list
            duration = float(len(idx_list)) / float(self.max_samples)
            offset = random.randint(0, int(duration))
            new_idx_list = []
            for i in range(self.max_samples):
                if int(i * duration + offset) < len(idx_list):
                    new_idx_list.append(int(i * duration + offset))
                else:
                    continue

            return new_idx_list

        filter_idx = _frame_filter(list(range(int((t_end - t_start) * 30))))
        poses_idx = self.index_list[video_name]["pose"]
        rgb_info_list = []
        poses = []
        for i in filter_idx:
            _i = int(t_start * 30) + i
            if _i < len(poses_idx):
                pos = poses_idx[_i]
                poses.append(pos)

        with TextFile(self.file_list[video_name]["pose"], blocking=False) as fp:
            lines = self.file_obj.seek_poses(poses, fp)
        for line in lines:
            line = line.strip().split("\t")
            line = [float(x) for x in line]
            rgb_info_list.append(line[2:])
        rgb_info_arr = np.array(rgb_info_list)  # equivalent to sample_dict['rgb-info']
        cam_poses = rgb_info_arr[:, :16]
        cam_poses = np.nan_to_num(cam_poses, copy=True, nan=0.0)

        for task in self.tasks:
            if "hands" in task:
                idx = self.index_list[video_name][task]
                hand_poses = []
                poses = []
                for i in filter_idx:
                    _i = int(t_start * 30) + i
                    if _i < len(idx):
                        pos = idx[int(t_start * 30) + i]
                        poses.append(pos)
                with TextFile(self.file_list[video_name][task], blocking=False) as fp:
                    lines = self.file_obj.seek_poses(poses, fp)
                for line in lines:
                    line = line.strip().split("\t")
                    line = [float(x) for x in line]
                    line_arr = np.array(line[2:])
                    if not np.isnan(np.sum(line_arr)):
                        # filtering out nan values
                        hand_poses.append(line[2:])

                hand_poses = np.array(hand_poses)
                if not self.hand_norm:
                    sample_dict[task] = hand_poses
                else:  # defaut is using hand normalization
                    hand_norm_list = []
                    for ii in range(hand_poses.shape[0]):
                        hand_poses_reshape = np.reshape(
                            hand_poses[ii][1:-52], (-1, 4, 4)
                        )
                        # To calculate relative hand pose to the cam
                        cam_pose_i_inv = np.linalg.pinv(cam_poses[ii].reshape(4, 4))
                        # assert not np.isnan(cam_pose_i_inv).any(), "cam_pose_i_inv is nan"  # noqa: E501
                        hand_norm = np.einsum(
                            "ij,njk->nik", cam_pose_i_inv, hand_poses_reshape
                        )
                        hand_norm = np.einsum(
                            "ij,njk->nik", self.axis_transform, hand_norm
                        )
                        hand_poses[ii][1:-52] = hand_norm.reshape(-1)
                        hand_norm_list.append(hand_poses[ii])
                    sample_dict[task] = np.array(hand_norm_list)

            elif "eye" in task:
                idx = self.index_list[video_name][task]
                gaze_poses = []
                poses = []
                for i in filter_idx:
                    _i = int(t_start * 30) + i
                    if _i < len(idx):
                        pos = idx[int(t_start * 30) + i]
                        poses.append(pos)
                with TextFile(self.file_list[video_name][task], blocking=False) as fp:
                    lines = self.file_obj.seek_poses(poses, fp)

                for line in lines:
                    line = line.strip().split("\t")
                    line = [float(x) for x in line]
                    line_arr = np.array(line[2:])
                    if not np.isnan(np.sum(line_arr)):
                        # filtering out nan values
                        gaze_poses.append(line[2:])

                gaze_poses = np.array(gaze_poses)
                gaze_poses = np.nan_to_num(gaze_poses, copy=True, nan=0.0)

                if not self.eye_norm:
                    sample_dict[task] = gaze_poses
                else:
                    eye_norm_list = []
                    for ii in range(gaze_poses.shape[0]):
                        eye_poses_reshape = gaze_poses[ii]

                        eye_trans = eye_poses_reshape[0:3]
                        direction_vec = eye_poses_reshape[3:6]
                        # To calculate relative eye 3D loc to the cam
                        cam_pose_i_inv = np.linalg.pinv(cam_poses[ii].reshape(4, 4))
                        eye_transformed = np.dot(
                            self.axis_transform,
                            np.dot(
                                cam_pose_i_inv,
                                np.concatenate((eye_trans, [1])),
                            ),
                        )
                        vec_transformed = np.dot(
                            self.axis_transform[:3, :3],
                            np.dot(
                                cam_pose_i_inv[:3, :3],
                                (direction_vec),
                            ),
                        )
                        # transformed_eye = np.ones(7)
                        eye_poses_reshape[:3] = eye_transformed.reshape(-1)[:3]
                        eye_poses_reshape[3:6] = vec_transformed.reshape(-1)[:3]

                        eye_norm_list.append(eye_poses_reshape)
                    sample_dict[task] = np.array(eye_norm_list)
            elif task == "rgb":
                video = EncodedVideo.from_path(self.v_list[video_name])
                fps = video._container.streams.video[0].average_rate
                # rgb_filter_idx = [int(x * fps / 30) for x in filter_idx]
                video_value = video.get_clip(start_sec=t_start, end_sec=t_end)["video"]
                rgb_filter_idx = [
                    int(x * fps / 30)
                    for x in filter_idx
                    if int(x * fps / 30) < video_value.size(1)
                ]
                if video_value is None:
                    logger.info(
                        f"video_name = {video_name}, t_start = {t_start}, t_end = {t_end}, filter_idx = {filter_idx[0]}, len_rgb = {rgb_filter_idx[0]}"
                    )
                    exit()
                    # try:
                try:
                    sample_dict[task] = video_value[:, rgb_filter_idx]
                except:
                    logger.info(
                        f"index error: video_name = {video_name}, t_start = {t_start}, t_end = {t_end}, {video_value.size(0)}, {video_value.size(1)}, {max(rgb_filter_idx)}"
                    )
                    exit()
                sample_dict[task] = (
                    sample_dict[task]
                    .transpose(0, 1)
                    .transpose(1, 2)
                    .transpose(2, 3)
                    .numpy()
                    .astype(np.uint8)
                )

            elif task == "head":
                idx = self.index_list[video_name][task]
                sample_dict[task] = []
                poses = []
                for i in filter_idx:
                    _i = int(t_start * 30) + i
                    if _i < len(idx):
                        pos = idx[int(t_start * 30) + i]
                        poses.append(pos)

                with TextFile(self.file_list[video_name][task], blocking=False) as fp:
                    lines = self.file_obj.seek_poses(poses, fp)
                for line in lines:
                    line = line.strip().split("\t")
                    line = [float(x) for x in line]
                    line_arr = np.array(line[2:])
                    if not np.isnan(np.sum(line_arr)):
                        # filtering out nan values
                        sample_dict[task].append(line[2:])

                sample_dict[task] = np.array(sample_dict[task])
                sample_dict[task] = np.nan_to_num(sample_dict[task], copy=True, nan=0.0)
            else:  # need to add depth shortly
                raise NotImplementedError

        for key in sample_dict:
            if len(sample_dict[key]) == 0:
                if key == "rgb":
                    sample_dict[key] = np.zeros((self.max_samples, 256, 454, 3)).astype(
                        np.uint8
                    )
                    logger.info("bad image sequences, pad zero images...")
                elif "hands" in key:
                    sample_dict[key] = np.zeros((self.max_samples, 469)).astype(
                        np.float32
                    )
                    # raise NotImplementedError
                elif "eye" in key:
                    sample_dict[key] = np.zeros((self.max_samples, 7)).astype(
                        np.float32
                    )
                    # raise NotImplementedError
                elif "head" in key:
                    sample_dict[key] = np.zeros((self.max_samples, 16)).astype(
                        np.float32
                    )
                    # raise NotImplementedError
                elif key == "depth-aligned":
                    sample_dict[key] = np.zeros((self.max_samples, 256, 454)).astype(
                        np.uint8
                    )
                    logger.info("bad depth sequences, pad zero images...")
                elif key == "rgb-info":
                    sample_dict[key] = np.zeros((self.max_samples, 41)).astype(
                        np.float32
                    )
                    logger.info("bad rgb-info sequences, pad zero images...")
                else:
                    print("key: ", key)
                    raise NotImplementedError
            else:
                if sample_dict[key].shape[0] < self.max_samples:
                    pads = [(0, self.max_samples - sample_dict[key].shape[0])]
                    if len(sample_dict[key].shape) > 1:
                        pads += [(0, 0)] * (len(sample_dict[key].shape) - 1)
                    sample_dict[key] = np.pad(sample_dict[key], pads, "constant")

        if self.transform is not None:
            for key in sample_dict:
                if len(sample_dict[key]) == 0:
                    logger.info(
                        "video name: {}, key: {}, cuda device: {}".format(
                            video_name, key, torch.cuda.current_device()
                        )
                    )
            sample_dict = self.transform(sample_dict)

        return sample_dict


# The code below can be used for debugging purposes. 

# if __name__ == "__main__":
#     from .utils import DataAugmentationMD

#     prefixes = {"rgb": ""}
#     transform = DataAugmentationMD(is_train=True)

#     dataset = MultiRawDataset(
#         root="",
#         label_root="",
#         tasks=["rgb", "hands-left", "hands-right", "eye", "head"],
#         subset="train",
#         load_type="recognition",  # choice = ['recogtnion', 'anticipation']
#         benchmark="fine_grained_action",
#         max_samples=16,
#         transform=transform,
#         target_transform=None,
#     )

#     sample_dict = dataset.__getitem__(1)
#     import pdb

#     pdb.set_trace()
