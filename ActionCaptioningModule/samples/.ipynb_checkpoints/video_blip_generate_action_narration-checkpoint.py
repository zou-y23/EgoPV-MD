import argparse

from pytorchvideo.data.video import VideoPathHandler
from transformers import Blip2Processor, PreTrainedTokenizerFast

from eilev.model.utils import process
from eilev.model.v1 import VideoBlipForConditionalGeneration
import json
from models.lta_models import *


def generate(
    model: VideoBlipForConditionalGeneration,
    processor: Blip2Processor,
    video_path: str,
    prompt: str,
) -> None:
    
    video_path_handler = VideoPathHandler()
    # process only the first 10 seconds
    # clip = video_path_handler.video_from_path(video_path).get_clip(0, 10)

    # sample a frame every 30 frames, i.e. 1 fps. We assume the video is 30 fps for now.
    # frames = clip["video"][:, ::30, ...].unsqueeze(0)

    # inputs = process(processor, video=frames, text=prompt.strip()).to(model.device)



    # 打开JSON文件
    with open('/data/t618141/data-annnotation-trainval-v1_1.json', 'r') as file:
        # 加载JSON数据
        data = json.load(file)
        for i in range(len(data)):
            # 现在你可以访问JSON数据中的任何部分
            # 例如，打印视频的分辨率
            
            name = data[i]['video_name']
            if name == 'R005-7July-GoPro':

                # 遍历事件并打印每个事件的标签和开始时间
                for event in data[i]['events']:
                    if event['label'] == 'Fine grained action':
                        video_path = '/data/t618141/EILEV-main/video/' + name + '/Video_compress.mp4'
                        clip = video_path_handler.video_from_path(video_path).get_clip(event['start'], event['end'])
                        print(clip["video"].shape)

                        frames = clip["video"][:, ::10, ...].unsqueeze(0)
                        print(frames.shape)
                        inputs = process(processor, video=frames, text=prompt.strip()).to(model.device)



                        generated_ids = model.generate(
                            **inputs,
                            num_beams=4,
                            max_new_tokens=128,
                            temperature=0.7,
                            top_p=0.9,
                            repetition_penalty=1.5,
                            do_sample=True,
                        )
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
                            0
                        ].strip()
                        print(f"ID: {event['id']}, start: {event['start']}, end: {event['end']}, Generated_text: {generated_text}, Verb: {event['attributes']['Verb']}, Noun: {event['attributes']['Noun']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an action narration using VideoBLIP."
    )
    parser.add_argument("video")
    parser.add_argument("prompt")
    parser.add_argument("--model", default="/data/t618141/EILEV-main/video-blip-flan-t5-xl-ego4d")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ARM = MultiTaskMViT()
        # 加载检查点文件
    checkpoint = torch.load('/data/t618141/epoch=9-step=44559.ckpt').device()

    # 加载模型权重
    ARM.load_state_dict(checkpoint['state_dict'])  # 或 checkpoint 本身，取决于保存方式

    model = VideoBlipForConditionalGeneration.from_pretrained(args.model).to(
        args.device
    )
    if args.processor is None:
        args.processor = args.model
    processor = Blip2Processor.from_pretrained(args.processor)
    generate(model, processor, args.video, args.prompt)
