import argparse
import torch

from pytorchvideo.data.video import VideoPathHandler
from transformers import Blip2Processor, PreTrainedTokenizerFast

from eilev.model.utils import process
from eilev.model.v1 import VideoBlipForConditionalGeneration
import json
from models.lta_models import *


def generate(
    model: VideoBlipForConditionalGeneration,
    processor: Blip2Processor,
    prompt: str,
    batch_size: int = 8,
) -> None:
    
    video_path_handler = VideoPathHandler()
    
    # Store all generated results
    results = []
    
    # Open JSON file
    with open('/root/autodl-tmp/data/annotations/data-annnotation-trainval-v1_1.json', 'r') as file:
        # Load JSON data
        data = json.load(file)
        total_processed = 0
        total_skipped = 0
        
        # Collect all events to process
        events_to_process = []
        max_events = 10000  # Limit number of events to process
        for i in range(len(data)):
            name = data[i]['video_name']
            video_path = '/root/autodl-tmp/data/clips/' + name + '/Export_py/Video_compress.mp4'
            print(f"Processing video {i}, collected {len(events_to_process)} events so far")
            
            for event in data[i]['events']:
                if event['label'] == 'Fine grained action':
                    events_to_process.append({
                        'event': event,
                        'video_path': video_path,
                        'video_name': name
                    })
                    # Stop collecting if enough events have been collected
                    if len(events_to_process) >= max_events:
                        break
            
            # Stop iterating through videos if enough events have been collected
            if len(events_to_process) >= max_events:
                break
        
        print(f"Total events collected: {len(events_to_process)}")
        
        # Process events in batches
        for batch_start in range(0, len(events_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(events_to_process))
            batch_events = events_to_process[batch_start:batch_end]
            
            # Prepare batch data
            batch_frames = []
            batch_event_info = []
            
            for event_info in batch_events:
                try:
                    clip = video_path_handler.video_from_path(event_info['video_path']).get_clip(
                        event_info['event']['start'], 
                        event_info['event']['end']
                    )
                    frames = clip["video"][:, ::10, ...]  # shape: (C, T, H, W)
                    batch_frames.append(frames)
                    batch_event_info.append(event_info)
                except Exception as e:
                    total_skipped += 1
                    print(f"Skipped event {event_info['event']['id']} from video {event_info['video_name']}: {str(e)}")
                    continue
            
            if len(batch_frames) == 0:
                continue
            
            # Find max time dimension and pad
            max_time = max(frames.shape[1] for frames in batch_frames)
            padded_frames = []
            for frames in batch_frames:
                # frames shape: (C, T, H, W)
                current_time = frames.shape[1]
                if current_time < max_time:
                    # Pad using the last frame
                    padding = frames[:, -1:, ...].repeat(1, max_time - current_time, 1, 1)
                    padded_frame = torch.cat([frames, padding], dim=1)
                else:
                    padded_frame = frames
                padded_frames.append(padded_frame.unsqueeze(0))  # Add batch dimension: (1, C, T, H, W)
            
            # Merge into batch
            batch_video = torch.cat(padded_frames, dim=0)  # (batch_size, C, T, H, W)
            
            # Prepare batch text prompts
            batch_prompts = [prompt.strip()] * len(batch_frames)
            
            try:
                # Batch processing
                inputs = process(processor, video=batch_video, text=batch_prompts)
                # Move inputs to correct device
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                # Batch generation
                generated_ids = model.generate(
                    **inputs,
                    num_beams=4,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    do_sample=True,
                )
                
                # Batch decode
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Process results in each batch
                for generated_text, event_info in zip(generated_texts, batch_event_info):
                    generated_text = generated_text.strip()
                    
                    verb = event_info['event']['attributes'].get('Verb', '')
                    noun = event_info['event']['attributes'].get('Noun', '')
                    
                    # Save result information
                    results.append({
                        'event_id': event_info['event']['id'],
                        'video_name': event_info['video_name'],
                        'start': event_info['event']['start'],
                        'end': event_info['event']['end'],
                        'reference_verb': verb,
                        'reference_noun': noun,
                        'generated_text': generated_text
                    })
                    
                    total_processed += 1
                    
                    if total_processed % 10 == 0:
                        print(f"Processed {total_processed} events...")
                        
            except Exception as e:
                # If batch processing fails, fall back to individual processing
                print(f"Batch processing failed: {str(e)}, falling back to individual processing")
                for event_info in batch_event_info:
                    try:
                        clip = video_path_handler.video_from_path(event_info['video_path']).get_clip(
                            event_info['event']['start'], 
                            event_info['event']['end']
                        )
                        frames = clip["video"][:, ::10, ...].unsqueeze(0)
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
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                        
                        verb = event_info['event']['attributes'].get('Verb', '')
                        noun = event_info['event']['attributes'].get('Noun', '')
                        
                        # Save result information
                        results.append({
                            'event_id': event_info['event']['id'],
                            'video_name': event_info['video_name'],
                            'start': event_info['event']['start'],
                            'end': event_info['event']['end'],
                            'reference_verb': verb,
                            'reference_noun': noun,
                            'generated_text': generated_text
                        })
                        
                        total_processed += 1
                        
                        if total_processed % 10 == 0:
                            print(f"Processed {total_processed} events...")
                            
                    except Exception as e2:
                        total_skipped += 1
                        print(f"Skipped event {event_info['event']['id']} from video {event_info['video_name']}: {str(e2)}")
                        continue
        
        # Output statistics
        if total_processed > 0:
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS")
            print("=" * 80)
            print(f"Total processed events: {total_processed}")
            print(f"Total skipped events: {total_skipped}")
            print(f"Batch size: {batch_size}")
            print("=" * 80)
            
            # Save results to JSON file
            output_file = '/root/autodl-tmp/eilev_results_10000.json'
            output_data = {
                'summary': {
                    'total_processed': total_processed,
                    'total_skipped': total_skipped,
                    'batch_size': batch_size
                },
                'results': results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to: {output_file}")
        else:
            print("No events were successfully processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an action narration using VideoBLIP."
    )
    parser.add_argument("--model", default="/root/autodl-tmp/video-blip-flan-t5-xl-ego4d")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for processing videos")
    args = parser.parse_args()

#     ARM = MultiTaskMViT()
#         # Load checkpoint file
#     checkpoint = torch.load('/data/t618141/epoch=9-step=44559.ckpt').device()

#     # Load model weights
#     ARM.load_state_dict(checkpoint['state_dict'])  

    model = VideoBlipForConditionalGeneration.from_pretrained(args.model).to(
        args.device
    )
    if args.processor is None:
        args.processor = args.model
    processor = Blip2Processor.from_pretrained(args.processor)
    args.prompt = 'Generate a sentence describing what the operator is doing based on the hand movements in this video.'
    generate(model, processor, args.prompt, batch_size=args.batch_size)
