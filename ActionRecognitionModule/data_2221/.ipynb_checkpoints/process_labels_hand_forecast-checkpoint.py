import json
from collections import OrderedDict, Counter
import random
import argparse
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='demo microgesture')
    parser.add_argument('--data_version', '-v', default="0724",
                        help='data version')
    parser.add_argument('--data_path', '-d', default="/mnt/fastdata/all-data-121922",
                        help='data path')
    parser.add_argument('--hand_forecast', '-f', action="store_true",
                        help='Keyboard mode')

    args = parser.parse_args()
    date_version = args.data_version
    labels = OrderedDict(
        {
            "fine_grained_action": OrderedDict(),
            "coarse_grained_action": OrderedDict(),
            "intervention_prediction_1": OrderedDict(),  # predict the type of the intervention given a window of 1, 3, 5s
            "intervention_prediction_3": OrderedDict(),
            "intervention_prediction_5": OrderedDict(),
            "intervention_detection_1": OrderedDict(),  # binary prediction for sliding window with 1s
            "intervention_detection_3": OrderedDict(),
            "intervention_detection_5": OrderedDict(),
            "mistake_prediction": OrderedDict(),
        }
    )
    label2idx = OrderedDict(
        {
            "fine_grained_action": [],
            "fine_grained_verb": [],
            "fine_grained_noun": [],
            "fine_grained_correctness": [],
            "mistake_prediction": [],
            "fine_grained_intervention_type": [],
            "task_type": [],
            "coarse_grained_action": [],
            "coarse_grained_verb": [],
            "coarse_grained_noun": [],
            "intervention_prediction": [],
            "intervention_detection": ["No Intervention", "Intervention"],
        }
    )

    per_task_videos = OrderedDict()

    label_file = 'data-annotation-trainval-v1_1.json'
    with open(label_file, "r") as fp:
        raw_labels = json.load(fp)

    # get list of videos in latest data splits
    all_vids = []
    with open('train-v1_2.txt', "r") as f:
        all_vids += f.read().splitlines()
    with open('val-v1_2.txt', "r") as f:
        all_vids += f.read().splitlines()
    with open('test-v1_2.txt', "r") as f:
        all_vids += f.read().splitlines()

    cnt_coarse = 0
    cnt_no_coarse = 0
    cnt_ignore_conv = 0
    cnt_conv = 0
    cnt_repeating_window = 0
    cnt_total_window = 0

    fine_action_counter = Counter([])
    coarse_action_counter = Counter([])

    for i, item in enumerate(raw_labels):
        # video_name = item["videoUrl"].split("/")[4]
        # key "videoUrl" is outdated
        video_name = item["video_name"]
        # skip any videos present in label but not in split
        if video_name not in all_vids:
            print(f"{video_name} not found in data split")
            continue
        if "%" in video_name:
            video_name = video_name.replace("%20", " ")
            print(video_name)
        if "taskType" not in item:
            task_type = "Not found"
            print(f"{video_name} not found in labels")
        else:
            task_type = item["taskType"]
        if task_type in label2idx["task_type"]:
            task_id = label2idx["task_type"].index(task_type)
        else:
            task_id = len(label2idx["task_type"])
            label2idx["task_type"].append(task_type)

        if args.hand_forecast:
            with open(os.path.join(args.data_path,video_name,'Export_py','Hands','Left_sync.txt')) as f:
                left_sync = f.readlines()
            with open(os.path.join(args.data_path,video_name,'Export_py','Hands','Right_sync.txt')) as f:
                right_sync = f.readlines()
            left_len = len(left_sync)
            right_len = len(right_sync)

        # add video_names_per_task
        if task_type not in per_task_videos:
            per_task_videos[task_type] = []
        per_task_videos[task_type].append(video_name)

        events = item["events"]

        # collect the coarse_grained_action time_stamps
        coarse_times = []
        for j, event in enumerate(events):
            if not "Coarse grained action" in event["label"]:
                continue
            coarse_times.append((float(event["start"]), float(event["end"])))
        sorted(coarse_times, key=lambda x: x[0])
        # print(video_name, "coarse action numbers: ", len(coarse_times))

        # collect all the start time of the intervention moment
        interv_times = []
        for j, event in enumerate(events):
            if not "Conversation" in event["label"]:
                continue
            conv_type = event["attributes"]["Conversation Purpose"]
            if (
                "other" in conv_type
                or "adjusting" in conv_type
                or "describing high-level instruction" in conv_type
                or "remarks" in conv_type
                or "ask questions" in conv_type
                or "instructor-reply-to-student" in conv_type
                or "student-start-conversation" in conv_type
            ):
                continue
            startTime = event["start"]
            interv_times.append(float(startTime))
        sorted(interv_times)

        for j, event in enumerate(events):
            if not "action" in event["label"]:
                continue
            startTime = event["start"]  # scale AI annotation used startTime and endTime
            endTime = event["end"]
            label_type = "_".join(event["label"].strip().lower().split(" "))
            verb = event["attributes"]["Verb"]
            noun = event["attributes"]["Noun"]
            if isinstance(verb, list):
                verb = verb[0]
            if isinstance(noun, list):
                noun = noun[0]

            verb = verb.lower()
            noun = noun.lower()
            action = verb + "-" + noun

            if action in label2idx[label_type]:
                action_id = label2idx[label_type].index(action)
            else:
                action_id = len(label2idx[label_type])
                label2idx[label_type].append(action)

            prefix = label_type.split("_")[:-1]
            verb_label = "_".join(prefix + ["verb"])
            noun_label = "_".join(prefix + ["noun"])
            if verb in label2idx[verb_label]:
                verb_id = label2idx[verb_label].index(verb)
            else:
                verb_id = len(label2idx[verb_label])
                label2idx[verb_label].append(verb)

            if noun in label2idx[noun_label]:
                noun_id = label2idx[noun_label].index(noun)
            else:
                noun_id = len(label2idx[noun_label])
                label2idx[noun_label].append(noun)

            if not video_name in labels[label_type]:
                labels[label_type][video_name] = []

            # adding fine-grained action correctness labels
            if "fine_grained" in label_type:
                fine_action_counter.update([action])
                # add correctness labels
                correctness = event["attributes"]["Action Correctness"]
                corr_type = "fine_grained_correctness"
                if correctness != "Correct Action":
                    correctness = "Wrong Action"  # otherwise, Wrong Action, ..., are all considered as Wrong Action

                if correctness in label2idx[corr_type]:
                    corr_id = label2idx[corr_type].index(correctness)
                else:
                    corr_id = len(label2idx[corr_type])
                    label2idx[corr_type].append(correctness)
                if args.hand_forecast:
                    left_pose = []
                    
                    for ii in range(45):
                        l = int(endTime * 30) + ii
                        line = left_sync[l] if l < left_len else left_sync[-1]
                        #line = left_sync[int(endTime * 30) + ii]
                        line = line.strip().split("\t")
                        line = [float(x) for x in line]
                        left_pose.append(line[2:])

                    right_pose = []
                    
                    for ii in range(45):
                        l = int(endTime * 30) + ii
                        line = right_sync[l] if l < right_len else right_sync[-1]
                        #line = right_sync[int(endTime * 30) + ii]
                        line = line.strip().split("\t")
                        line = [float(x) for x in line]
                        right_pose.append(line[2:])


                    labels[label_type][video_name].append(
                        [
                            startTime,
                            endTime,
                            [task_id, action_id, verb_id, noun_id, corr_id],
                            {"left":left_pose,"right":right_pose},
                        ]
                    )

                else:
                    labels[label_type][video_name].append(
                        [
                            startTime,
                            endTime,
                            [task_id, action_id, verb_id, noun_id, corr_id],
                        ]
                    )
                # add the mistake detection clips where the start time
                # is the start time of the coarse grained action and end time
                # of the end of the fine-grained action
                mistake_start = startTime  # default is the fine-grained action start
                for s, e in coarse_times:
                    if s <= startTime <= e and s <= endTime <= e:
                        mistake_start = s
                        break
                if mistake_start == startTime:
                    cnt_no_coarse += 1
                else:
                    cnt_coarse += 1
                    mistake_type = "mistake_prediction"
                    # adding to the label list only when the fine-grained action is within the coarse grained action
                    if not video_name in labels[mistake_type]:
                        labels[mistake_type][video_name] = []
                    labels[mistake_type][video_name].append(
                        [mistake_start, endTime, [task_id, corr_id]]
                    )
            else:
                coarse_action_counter.update([action])
                labels[label_type][video_name].append(
                    [
                        startTime,
                        endTime,
                        [task_id, action_id, verb_id, noun_id],
                    ]
                )
        # process conversation
        for j, event in enumerate(events):
            if not "Conversation" in event["label"]:
                continue
            startTime = event["start"]
            endTime = event["end"]

            conv_type = event["attributes"]["Conversation Purpose"]
            if (
                "other" in conv_type
                or "adjusting" in conv_type
                or "describing high-level instruction" in conv_type
                or "remarks" in conv_type
                or "ask questions" in conv_type
                or "instructor-reply-to-student" in conv_type
                or "student-start-conversation" in conv_type
            ):
                continue

            coarse_start_time = 0
            coarse_end_time = 0
            for s, e in coarse_times:
                if s <= startTime <= e and s <= endTime <= e:
                    coarse_start_time = s
                    coarse_end_time = e
                    break
            if coarse_start_time == 0 or coarse_end_time == 0:
                # print("ignore conversations not in the coarse_grained_action")
                cnt_ignore_conv += 1
                continue
            cnt_conv += 1
            label_type = "intervention_prediction"
            if conv_type in label2idx[label_type]:
                conv_id = label2idx[label_type].index(conv_type)
            else:
                conv_id = len(label2idx[label_type])
                label2idx[label_type].append(conv_type)

            # update timestamps
            for time_range in [1, 3, 5]:
                prev_time = max(coarse_start_time, startTime - time_range)
                sub_label_type = label_type + f"_{time_range}"
                detection_type = f"intervention_detection_{time_range}"

                if not video_name in labels[sub_label_type]:
                    labels[sub_label_type][video_name] = []

                if not video_name in labels[detection_type]:
                    labels[detection_type][video_name] = []

                labels[sub_label_type][video_name].append(
                    [prev_time, startTime, [task_id, conv_id]]
                )
                # adding binary prediction window for intervention detection
                start = coarse_start_time
                while start + time_range < prev_time:  # before the intervention happens
                    l = 0
                    end = start + time_range
                    for e in interv_times:
                        if abs(end - e) < 0.5:
                            l = 1
                            # print("repeating window appeared")
                            cnt_repeating_window += 1
                            break
                    # dealing with multiple intervention within a coarse grained action
                    if [start, start + time_range, [task_id, l]] not in labels[
                        detection_type
                    ][video_name]:
                        labels[detection_type][video_name].append(
                            [start, start + time_range, [task_id, l]]
                        )
                        cnt_total_window += 1
                    start += time_range
                # adding the intervention happens
                labels[detection_type][video_name].append(
                    [prev_time, startTime, [task_id, 1]]
                )
                cnt_total_window += 1

    label2idx["mistake_prediction"] = label2idx["fine_grained_correctness"]
    # print stats
    for k, v in label2idx.items():
        print(f"label2idx\t\t{k}\t{len(v)}")

    print(
        f"Stats: {cnt_no_coarse} actions are not within a coarse grained action and {cnt_coarse} are in a coarseg grained action"
    )

    print(
        f"Stats: {cnt_ignore_conv} convs are ignored since not in a coarse grained action and {cnt_conv} are included"
    )

    print(
        f"Stats: {cnt_repeating_window} windows are conflicting with other interventions and {cnt_total_window} are total windows"
    )

    with open(f"labels_{date_version}_2221_classes.json", "w") as fp:
        json.dump(labels, fp)

    with open(f"labels_{date_version}_2221_label2idx.json", "w") as fp:
        json.dump(label2idx, fp)

    # obtain the head classes and save their label2idx
    label2idx_head = OrderedDict(label2idx)
    label2idx_head["fine_grained_action"] = [
        value for value, count in fine_action_counter.items() if count > 10
    ]
    label2idx_head["coarse_grained_action"] = [
        value for value, count in coarse_action_counter.items() if count > 10
    ]
    print(
        f'{len(label2idx_head["fine_grained_action"])} head fine grained actions and {len(label2idx_head["coarse_grained_action"])} head coarse grained actions'
    )

    with open(f"labels_{date_version}_2221_label2idx_head_above10.json", "w") as fp:
        json.dump(label2idx_head, fp)
