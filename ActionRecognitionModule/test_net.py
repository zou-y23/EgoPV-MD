"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
from einops import rearrange
import datetime
import json
import os
import os.path as osp

from models.VideoEncoder.videoencoder.utils import checkpoint as cu
from models.VideoEncoder.videoencoder.utils import distributed as du
from models.VideoEncoder.videoencoder.utils import logging as logging
from models.VideoEncoder.videoencoder.utils import misc as misc
from models.VideoEncoder.videoencoder.utils import metrics as metrics
from models.VideoEncoder.videoencoder.visualization import tensorboard_vis as tb
from models.seq2seq.models.build import build_model as build_model_seq
from models.VideoEncoder.videoencoder.models import build_model
from fvcore.common.timer import Timer

from torchmetrics.functional import precision_recall
from data_loader import data_loader as loader
from models.vit_mdn import vit_base_patch16_224
from train_net import create_mask

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    tasks = cfg.TASKS
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    if cfg.DATA.BENCHMARK == "hand_forecast":
        epoch_dists = [0 for x in range(len(cfg.DATA.EVAL_LEN))]
        hand_xyz = []
        for kk in range(26):
            hand_xyz += [4 + kk * 16, 8 + kk * 16, 12 + kk * 16]
        hand_xyz = torch.Tensor(hand_xyz).to(torch.int64)

    cnt = 0
    for cur_iter, (data, labels) in enumerate(test_loader):
        if cfg.DATA.BENCHMARK == "hand_forecast":

            left_mask_eval, right_mask_eval = create_mask(labels, hand_xyz ,device, cfg.DATA.CLIP_DIST)
            
            preds = model(
                data ,labels, cfg.TASKS, teacher_forcing_ratio=0,
            )
            test_meter.data_toc()
            preds = preds.reshape(
                -1, cfg.DATA.NUM_EVAL_FRAMES, cfg.DATA.DIM_NUM * 2
            )
            preds_left = preds[:, :, : cfg.DATA.DIM_NUM]
            preds_right = preds[:, :, cfg.DATA.DIM_NUM :]
            trg_left =  labels["hands-left"].index_select(2, hand_xyz).to(device).float()
            trg_right = labels["hands-right"].index_select(2, hand_xyz).to(device).float()

        else:
            inputs = {}
            for task in tasks:
                if task == "rgb" or "depth" in task:
                    x = data[task].to(device).float()
                    x = rearrange(x, "b t h w c -> b c t h w")
                else:
                    x = data[task].to(device).float()
                inputs[task] = x

            # Transfer the data to the current GPU device.
            labels = labels.to(device)
            test_meter.data_toc()

            # Perform the forward pass.
            preds = model(**inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.DATA.BENCHMARK not in "hand_forecast":
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        if cfg.DATA.BENCHMARK == "hand_forecast":
            test_meter.update_stats_handforecast(
                preds_left.detach(),
                trg_left.detach(),
                left_mask_eval.detach(),
                preds_right.detach(),
                trg_right.detach(),
                right_mask_eval.detach(),
                cnt,
            )
        else:
            test_meter.update_stats(preds.detach(), labels.detach(), cnt)
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()
        cnt += preds.size(0)

    # get finalized evaluation metrics
    if cfg.MODEL.NUM_CLASSES <= 10:
        ks = (1,)
    elif cfg.DATA.BENCHMARK in "hand_forecast":
        ks = (15, 30, 45)
    else:
        ks = (1, 2)
    test_meter.finalize_metrics(ks=ks)

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        # if cfg.TEST.SAVE_RESULTS_PATH != "":
        save_path = os.path.join(cfg.OUTPUT_DIR, "test_preds.pkl")
        with PathManager.open(save_path, "wb") as fp:
            pickle.dump([all_preds, all_labels], fp)

        # save logs
        save_log = os.path.join(cfg.OUTPUT_DIR, "test_metrics.json")
        with open(save_log, "w") as fp:
            json.dump(test_meter.stats, fp)

        print(test_meter.stats)

        logger.info("Successfully saved prediction results to {}".format(save_path))
        # easy copy
        if cfg.DATA.BENCHMARK in ["fine_grained_action", "coarse_grained_action"]:
            cls_string = [
                "all_action",
                "all_verb",
                "all_noun",
                "head_action",
                "head_verb",
                "head_noun",
                "tail_action",
                "tail_verb",
                "tail_noun",
            ]
            cls_string = ",".join(cls_string)

            keys = [
                ["all_top1_acc", "all_top5_acc"],
                ["all_top1_acc_verb", "all_top5_acc_verb"],
                ["all_top1_acc_noun", "all_top5_acc_noun"],
                ["head_top1_acc", "head_top5_acc"],
                ["head_top1_acc_verb", "head_top5_acc_verb"],
                ["head_top1_acc_noun", "head_top5_acc_noun"],
                ["tail_top1_acc", "tail_top5_acc"],
                ["tail_top1_acc_verb", "tail_top5_acc_verb"],
                ["tail_top1_acc_noun", "tail_top5_acc_noun"],
            ]

            value = [
                "{}/{}".format(test_meter.stats[top1], test_meter.stats[top5])
                for top1, top5 in keys
            ]
            value = ",".join(value)
            logger.info(cls_string)
            logger.info(value)
        elif cfg.DATA.BENCHMARK == "mistake_prediction":
            keys = [
                "all_top1_acc",
                "all_top1_prec",
                "all_top1_recall",
                "correct_top1_prec",
                "correct_top1_recall",
                "wrong_top1_prec",
                "wrong_top1_recall",
            ]
            cls_string = ",".join(keys)
            value = ["{}".format(test_meter.stats[top1]) for top1 in keys]
            value = ",".join(value)
            logger.info(cls_string)
            logger.info(value)
        else:
            keys = sorted(list(test_meter.stats.keys()))
            cls_string = ",".join(keys)
            value = ["{}".format(test_meter.stats[top1]) for top1 in keys]
            value = ",".join(value)
            logger.info(cls_string)
            logger.info(value)

    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
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

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    if cfg.DATA.BENCHMARK == "hand_forecast":
        model = build_model_seq(cfg)
    else:
        model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Create meters for multi-view testing.
    class_file = osp.join(cfg.LABEL_DIR, cfg.DATA.CLASS_FILE)
    head_class_file = osp.join(cfg.LABEL_DIR, cfg.DATA.HEAD_CLASS_FILE)

    test_meter = MyTestMeter(
        len(test_loader.dataset),
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        class_file,
        head_class_file,
        cfg.DATA.BENCHMARK,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()


# define customized evaluation code for TestMeter
class MyTestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,  # number of action segments loaded
        num_cls,
        overall_iters,
        label2idx_file,
        label2idx_head_file,
        benchmark="fine_grained_action",
    ):
        """
        Construct tensors to store the predictions and labels.
        Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
            label2idx_file (str): path to the full label2idx file
            label2idx_head_file (str): path to the label2idx file of the head categories. `
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.overall_iters = overall_iters
        self.num_cls = num_cls
        self.ensemble_method = "sum"
        self.benchmark = benchmark
        self.eval_len = 45
        self.dim_size = 78  # 78*2
        # Initialize tensors.
        if self.benchmark in "hand_forecast":
            # need to store all the hands poses and prediction.
            self.left_preds = torch.zeros(
                (num_videos, self.eval_len, self.dim_size)
            )
            self.left_trgs = torch.zeros(
                (num_videos, self.eval_len, self.dim_size)
            )
            self.left_masks = torch.zeros((num_videos, self.eval_len)).bool()

            self.right_preds = torch.zeros(
                (num_videos, self.eval_len, self.dim_size)
            )
            self.right_trgs = torch.zeros(
                (num_videos, self.eval_len, self.dim_size)
            )
            self.right_masks = torch.zeros((num_videos, self.eval_len)).bool()
        
        self.video_preds = torch.zeros((num_videos, num_cls))
        self.video_labels = torch.zeros((num_videos)).long()
        # logger.info('num_videos: {}'.format(num_videos))
        self.scenario_count = torch.zeros(
            (num_videos)
        ).long()  # count number of actions per senario
        self.topk_accs = []
        self.stats = {}

        # loading the label2idx files
        with open(label2idx_file, "r") as fp:
            self.label2idx = json.load(fp)  # a dictionary of lists of label names

        with open(label2idx_head_file, "r") as fp:
            self.label2idx_head = json.load(fp)

        self.num_scenarios = len(self.label2idx["task_type"])
        self.scenarios = self.label2idx["task_type"]
        key = (
            "coarse_grained_action"
            if "coarse_grained" in benchmark
            else "fine_grained_action"
        )
        prefix = "coarse_grained_" if "coarse_grained" in benchmark else "fine_grained_"
        self.class_strings = self.label2idx[key]  # list of names of the actions
        self.verb_strings = self.label2idx[prefix + "verb"]  # list of verbs
        self.noun_strings = self.label2idx[prefix + "noun"]  # list of nouns

        self.class_strings_head = self.label2idx_head[key]
        self.head_ids = [self.class_strings.index(v) for v in self.class_strings_head]
        self.tail_ids = [
            i for i in range(len(self.class_strings)) if i not in self.head_ids
        ]

        if self.benchmark in ["fine_grained_action", "coarse_grained_action"]:
            self.verb_counts = torch.zeros((num_videos, 2))
            self.noun_counts = torch.zeros((num_videos, 2))

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.scenario_count.zero_()
        self.video_preds.zero_()
        self.video_labels.zero_()

        if self.benchmark in ["fine_grained_action", "coarse_grained_action"]:
            self.verb_counts.zero_()
            self.noun_counts.zero_()

        if self.benchmark in "hand_forecast":
            self.left_preds.zero_()
            self.left_trgs.zero_()
            self.left_masks.zero_()

            self.right_preds.zero_()
            self.right_trgs.zero_()
            self.right_masks.zero_()

    def update_stats_handforecast(
        self,
        left_preds,
        left_trgs,
        left_masks,
        right_preds,
        right_trgs,
        right_masks,
        global_idx,
    ):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            global_idx: start_id in the dataset
        """

        for ind in range(left_preds.shape[0]):
            vid_id = global_idx + ind
            self.left_preds[vid_id] = left_preds[ind]
            self.left_trgs[vid_id] = left_trgs[ind]
            self.left_masks[vid_id] = left_masks[ind]
            self.right_preds[vid_id] = right_preds[ind]
            self.right_trgs[vid_id] = right_trgs[ind]
            self.right_masks[vid_id] = right_masks[ind]

    def update_stats(self, preds, labels, global_idx):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            global_idx: start_id in the dataset
        """


        for ind in range(preds.shape[0]):
            vid_id = global_idx + ind
            if vid_id >= len(self.video_labels):
                continue
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            else:
                raise NotImplementedError

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def eulidean_dist(self, pred, label, mask):
        point_dist = (
            torch.abs(
                torch.sum(
                    (
                        pred[mask].reshape((-1, 3))
                        - label[mask].reshape((-1, 3))
                    )
                    ** 2,
                    1,
                )
            )
            ** 0.5
        )
        mean_dist = torch.mean(point_dist)
        return mean_dist

    def cal_eulidean_dists(self, pred, label, mask, ks):
        dists = []
        for len in ks:
            dists.append(
                self.eulidean_dist(
                    pred[:, :len], label[:, :len], mask[:, :len]
                )
            )
        return dists


    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        # Helper function to map k=2 to k=5 for display purposes only
        def display_k(k):
            return 5 if k == 2 else k

        self.stats = {"split": "test_final"}
        # prediction of all classes
        num_topks_correct = metrics.topks_correct(
            self.video_preds, self.video_labels, ks
        )
        topks = [(x / self.video_preds.size(0)) * 100.0 for x in num_topks_correct]

        if self.benchmark in "hand_forecast":
            left_num_topks_correct = self.cal_eulidean_dists(
                self.left_preds, self.left_trgs, self.left_masks, ks
            )
            right_num_topks_correct = self.cal_eulidean_dists(
                self.right_preds, self.right_trgs, self.right_masks, ks
            )

            for k, topk in zip(ks, left_num_topks_correct):
                self.stats["left {}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=6
                )
            for k, topk in zip(ks, right_num_topks_correct):
                self.stats["right {}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=6
                )
        elif self.benchmark in ["fine_grained_action", "coarse_grained_action"]:
            for k, topk in zip(ks, topks):
                self.stats["all_top{}_acc".format(display_k(k))] = "{:.{prec}f}".format(topk, prec=2)
            # prediction of head categories.
            mask_head = torch.any(
                torch.stack(
                    [
                        torch.eq(self.video_labels, aelem)
                        for aelem in self.head_ids
                    ],
                    dim=0,
                ),
                dim=0,
            )
            video_preds_head = self.video_preds[mask_head]
            video_labels_head = self.video_labels[mask_head]
            num_topks_correct_head = metrics.topks_correct(
                video_preds_head, video_labels_head, ks
            )
            topks_head = [
                (x / video_preds_head.size(0)) * 100.0 for x in num_topks_correct_head
            ]

            # prediction of tail categories.
            mask_tail = torch.any(
                torch.stack(
                    [
                        torch.eq(self.video_labels, aelem).logical_or_(
                            torch.eq(self.video_labels, aelem)
                        )
                        for aelem in self.tail_ids
                    ],
                    dim=0,
                ),
                dim=0,
            )
            video_preds_tail = self.video_preds[mask_tail]
            video_labels_tail = self.video_labels[mask_tail]
            num_topks_correct_tail = metrics.topks_correct(
                video_preds_tail, video_labels_tail, ks
            )
            topks_tail = [
                (x / video_preds_head.size(0)) * 100.0 for x in num_topks_correct_tail
            ]
            assert len({len(ks), len(topks)}) == 1
            topks_list = [topks_head, topks_tail]
            names = ["head", "tail"]

            for _name, _topks in zip(names, topks_list):
                for k, topk in zip(ks, _topks):
                    self.stats["{}_top{}_acc".format(_name, display_k(k))] = "{:.{prec}f}".format(
                        topk, prec=2
                    )

            # check verb and noun predictions for fine_grained or coarse_grained actions
            names = ["all", "head", "tail"]
            _top_max_k_vals, top_max_k_inds = torch.topk(
                self.video_preds, max(ks), dim=1, largest=True, sorted=True
            )
            pred_string = [
                [
                    self.class_strings[top_max_k_inds[i, j].long().item()]
                    for j in range(max(ks))
                ]
                for i in range(top_max_k_inds.size(0))
            ]
            gt_string = [self.class_strings[i.item()] for i in self.video_labels]

            for i, (pred_list, gt) in enumerate(zip(pred_string, gt_string)):
                gt_verb = gt.split("-")[0].strip()
                gt_noun = gt.split("-")[1].strip()
                pred_verbs = [x.split("-")[0].strip() for x in pred_list]
                pred_nouns = [x.split("-")[1].strip() for x in pred_list]
                if gt_verb == pred_verbs[0]:
                    self.verb_counts[i, 0] += 1
                if gt_verb in pred_verbs:
                    self.verb_counts[i, 1] += 1
                if gt_noun == pred_nouns[0]:
                    self.noun_counts[i, 0] += 1
                if gt_noun in pred_nouns:
                    self.noun_counts[i, 1] += 1

            verb_count_list = [
                self.verb_counts,
                self.verb_counts[mask_head],
                self.verb_counts[mask_tail],
            ]
            noun_count_list = [
                self.noun_counts,
                self.noun_counts[mask_head],
                self.noun_counts[mask_tail],
            ]

            for _cls, _counts in zip(
                ["verb", "noun"], [verb_count_list, noun_count_list]
            ):
                for _name, _c in zip(names, _counts):
                    accs = _c.mean(0) * 100.0
                    for k, topk in zip(ks, accs.tolist()):
                        self.stats[
                            "{}_top{}_acc_{}".format(_name, display_k(k), _cls)
                        ] = "{:.{prec}f}".format(topk, prec=2)

        elif self.benchmark == "mistake_prediction":
            for k, topk in zip(ks, topks):
                self.stats["all_top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)
            prec_recalls = calc_precision_recall(
                self.video_preds, self.video_labels, num_classes=self.num_cls, topks=ks
            )
            for k, (p, r) in zip(ks, prec_recalls):
                for i, _cls in enumerate(["correct", "wrong", "correction"]):
                    self.stats["{}_top{}_prec".format(_cls, k)] = "{:.{prec}f}".format(
                        p[i], prec=2
                    )
                    self.stats[
                        "{}_top{}_recall".format(_cls, k)
                    ] = "{:.{prec}f}".format(r[i], prec=2)
                self.stats["all_top{}_prec".format(k)] = "{:.{prec}f}".format(
                    np.mean(p), prec=2
                )
                self.stats["all_top{}_recall".format(k)] = "{:.{prec}f}".format(
                    np.mean(r), prec=2
                )

        elif "intervention_prediction" in self.benchmark:
            for k, topk in zip(ks, topks):
                self.stats["all_top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)
            prec_recalls = calc_precision_recall(
                self.video_preds,
                self.video_labels,
                num_classes=self.num_cls,
                topks=ks,
                average=None,
            )
            for k, (p, r) in zip(ks, prec_recalls):
                for i, _cls in enumerate(
                    ["follow-up", "correct-wrong-action", "confirming-prev-action"]
                ):
                    self.stats["{}_top{}_prec".format(_cls, k)] = "{:.{prec}f}".format(
                        p[i], prec=2
                    )
                    self.stats[
                        "{}_top{}_recall".format(_cls, k)
                    ] = "{:.{prec}f}".format(r[i], prec=2)
                self.stats["all_top{}_prec".format(k)] = "{:.{prec}f}".format(
                    np.mean(p), prec=2
                )
                self.stats["all_top{}_recall".format(k)] = "{:.{prec}f}".format(
                    np.mean(r), prec=2
                )
        elif "intervention_detection" in self.benchmark:
            for k, topk in zip(ks, topks):
                self.stats["all_top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)
            prec_recalls = calc_precision_recall(
                self.video_preds,
                self.video_labels,
                num_classes=self.num_cls,
                topks=ks,
                average=None,
            )
            for k, (p, r) in zip(ks, prec_recalls):
                for i, _cls in enumerate(["no-intervention", "intervention"]):
                    self.stats["{}_top{}_prec".format(_cls, k)] = "{:.{prec}f}".format(
                        p[i], prec=2
                    )
                    self.stats[
                        "{}_top{}_recall".format(_cls, k)
                    ] = "{:.{prec}f}".format(r[i], prec=2)
                self.stats["all_top{}_prec".format(k)] = "{:.{prec}f}".format(
                    np.mean(p), prec=2
                )
                self.stats["all_top{}_recall".format(k)] = "{:.{prec}f}".format(
                    np.mean(r), prec=2
                )

        logging.log_json_stats(self.stats)


def calc_precision_recall(preds, target, num_classes, topks, average=None):
    prec_recalls = []
    for topk in topks:
        prec, recall = precision_recall(
            preds, target, average=average, num_classes=num_classes, top_k=topk
        )
        prec_recalls.append([prec.numpy() * 100.0, recall.numpy() * 100.0])
    return prec_recalls
