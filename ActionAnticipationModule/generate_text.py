#!/usr/bin/env python3
"""
Feed data from prompt_for_gpt.json to Llama model continuously
Only use system and user content from messages
"""

import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm
import sys

# Try importing PEFT library (for loading LoRA models)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT library not installed, cannot load LoRA models. To use LoRA models, install: pip install peft")

# Fixed system content - no longer read from JSON file each time
SYSTEM_CONTENT = """

Example 1:
Narration: The student changes the battery for the GoPro.; Past actions: rotate gopro, pull battery_door, open battery_door, rotate gopro, grab battery, withdraw battery, rotate battery, place battery. Step 1: Confirm the task is changing the battery; sub-goals are exposing the battery compartment, removing the old battery, preparing the new battery, inserting it, and securing the door.
Step 2: The sequence shows the door opened and the battery grabbed, withdrawn, rotated, and placed aside - the old battery has been removed and set down.
Step 3: Typical pattern after placing a battery is to pick up or lift the replacement battery to insert it next.
Step 4: Final prediction - "lift battery". Next action: lift battery

Example 2:
Narration: The student opens the GoPro.; Past actions: press gopro, hold gopro, press gopro, hold gopro, rotate gopro, press gopro, rotate gopro, hold gopro. Step 1: Confirm the task is opening the GoPro; sub-goals include pressing latches/buttons and manipulating the housing (holding/rotating) to release a catch.
Step 2: Repeated press and hold actions interleaved with rotates indicate attempts to actuate a latch while adjusting orientation.
Step 3: The pattern implies the mechanism still needs a final press to release - pressing is the repeated, necessary triggering action.
Step 4: Final prediction - "press gopro". Next action: press gopro

Example 3:
Narration: The student inserts the battery and SD card into the Go Pro.; Past actions: grab battery, lift battery, insert battery, push battery, close battery_door, press battery, close battery_door, open battery_door. Step 1: Confirm the task is inserting battery and SD card; sub-goals are inserting/pushing the battery into place, closing the door, then opening it to access the SD slot.
Step 2: Actions show the battery was inserted and the door closed, then reopened, indicating preparation to insert the SD card after battery placement.
Step 3: After opening the battery door, the next immediate action is usually to take (grab) the SD card for insertion.
Step 4: Final prediction - "grab sd_card". Next action: grab sd_card

Example 4:
Narration: The student attaches the GoPro mount.; Past actions: grab gopro, grab screw, unscrew screw, withdraw screw, withdraw handheld_grip, place handheld_grip, grab strap, drop strap. Step 1: Confirm the task is attaching the GoPro mount; sub-goals include handling screws and mount components, preparing and aligning the mounting peg or bracket.
Step 2: The student removed a screw and handled the grip and strap, then dropped the strap-this indicates finishing preparatory steps and moving to alignment/inspection.
Step 3: The next logical step before final assembly is to inspect the mounting peg or receiving fixture to ensure alignment.
Step 4: Final prediction - "inspect mounting_peg". Next action: inspect mounting_peg

Example 5:
Narration: The student changes the battery.; Past actions: grab battery, lift battery, insert battery, withdraw battery, close battery_door, hold battery, open battery_door, insert battery. Step 1: Confirm the task is changing the battery; sub-goals are removing the old battery, inserting the new one, and closing/locking the battery door.
Step 2: The sequence shows an initial insert, withdrawal, door close/open, and another insert - the battery has just been inserted again.
Step 3: Once the battery is inserted, the standard next action is to secure the compartment by closing the battery door.
Step 4: Final prediction - "close battery_door". Next action: close battery_door

Example 6:
Narration: The student attaches the lens.; Past actions: rotate lens, rotate lens, withdraw lens, place lens, rotate lens, withdraw lens, inspect lens, place lens. Step 1: Confirm the task is attaching the lens; sub-goals include aligning the lens mount, placing the lens, and rotating/tightening to lock it in place.
Step 2: Multiple rotate and place/withdraw cycles indicate alignment attempts; inspect then place suggests the lens has been positioned.
Step 3: After placing a lens, a subsequent rotate is commonly performed to secure or tighten the mount.
Step 4: Final prediction - "rotate lens". Next action: rotate lens

Example 7:
Narration: The student installs the joy-con controllers back into the Nintendo Switch.; Past actions: lift nintendo_switch, flip joy_con_controller, insert joy_con_controller, withdraw joy_con_controller, rotate joy_con_controller, rotate nintendo_switch, insert joy_con_controller, flip joy_con_controller. Step 1: Confirm the task is reinstalling Joy-Con controllers; sub-goals include orienting the controller correctly, inserting it into the rail, and locking it into place.
Step 2: The sequence contains multiple inserts, withdrawals, rotations, and flips - the student is adjusting orientation for a proper fit.
Step 3: The latest action is a flip, which is often the final orientation adjustment before confirming insertion.
Step 4: Final prediction - "flip joy_con_controller". Next action: flip joy_con_controller

Example 8:
Narration: The student attaches the wheel to the wheel unit.; Past actions: hold wheel, slide wheel, hold wheel, hold wheel, hold wheel, turn wheel, hold wheel, place wheel. Step 1: Confirm the task is attaching the wheel; sub-goals include sliding the wheel onto the hub, aligning, rotating to seat it, and assembling/fastening.
Step 2: The student slid and turned the wheel and then placed it, which suggests alignment is done and the part is in position.
Step 3: After positioning, the typical next step is to perform final assembly or secure the wheel so it is fixed.
Step 4: Final prediction - "assemble wheel". Next action: assemble wheel

Example 9:
Narration: The student attaches the wheel to the wheel unit.; Past actions: place wheel, hold wheel, point wheel, grab wheel, rotate wheel, hold wheel, rotate wheel, rotate wheel. Step 1: Confirm the task is attaching the wheel; goals are positioning, aligning by rotation, and final placement/securement.
Step 2: Multiple rotates and holds indicate fine alignment is in progress after the wheel was grabbed and pointed into place.
Step 3: Once alignment rotations are complete, the natural next action is to set the wheel down into its final position.
Step 4: Final prediction - "place wheel". Next action: place wheel

Example 10:
Narration: The student makes a cup of coffee.; Past actions: push drip_tray, place cup, approach button, press button, hold cup, approach cup, grab cup, slide cup. Step 1: Confirm the task is making coffee; sub-goals include placing the cup under the dispenser, starting the machine, retrieving the cup, and placing or serving the cup.
Step 2: The student placed the cup, pressed the button to brew, then grabbed and slid the cup - indicating they are moving the finished cup.
Step 3: After sliding the cup, the next expected action is to put the cup down in its final spot.
Step 4: Final prediction - "place cup". Next action: place cup

I have the following action description and the past eight actions. Based on them and the example provided above, reason and predict the next action in a format of (verb, noun) pair."""

def load_model_and_tokenizer(model_path, lora_path=None):
    """
    Load model and tokenizer, supports LoRA fine-tuned models
    
    Args:
        model_path: Base model path
        lora_path: LoRA adapter path (optional, if provided will load LoRA weights)
    
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    import os as os_module
    
    # Check if it's a LoRA model directory
    if lora_path is None:
        # Check if model_path is a LoRA adapter directory
        adapter_config = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            print(f"🔍 Detected LoRA adapter directory: {model_path}")
            # Read base model path
            with open(adapter_config, 'r') as f:
                adapter_config_data = json.load(f)
                base_model_path = adapter_config_data.get("base_model_name_or_path")
                if base_model_path:
                    lora_path = model_path
                    model_path = base_model_path
                else:
                    print("⚠️  Cannot find base model path from adapter_config.json")
    
    # Temporarily modify config.json to use fixed values for loading
    config_path = os.path.join(model_path, "config.json")
    pid = os_module.getpid()
    original_config_path = config_path + f".orig_generate_text_{pid}"
    config_modified = False
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Always replace with fixed values for loading
            if config_dict.get('model_type') != 'llama':
                config_dict['model_type'] = 'llama'
                config_modified = True
            
            if 'rope_scaling' in config_dict and isinstance(config_dict['rope_scaling'], dict):
                if config_dict['rope_scaling'].get('rope_type') != 'llama3':
                    config_dict['rope_scaling']['rope_type'] = 'llama3'
                    config_modified = True
            
            if config_modified:
                # Save original config
                if os.path.exists(original_config_path):
                    os.remove(original_config_path)
                os.rename(config_path, original_config_path)
                
                # Write modified config
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # Load tokenizer (prefer loading from LoRA directory, otherwise from base model)
        tokenizer_path = lora_path if lora_path and os.path.exists(os.path.join(lora_path, "tokenizer.json")) else model_path
        print(f"📝 Loading tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load base model
        print("⏳ Loading base model (this may take some time)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    finally:
        # Restore original config.json
        if config_modified and os.path.exists(original_config_path):
            if os.path.exists(config_path):
                os.remove(config_path)
            os.rename(original_config_path, config_path)
    
    # If LoRA path is provided, load LoRA weights
    if lora_path:
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required to load LoRA models. Please install: pip install peft")
        
        print(f"🔗 Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print("✅ LoRA adapter loaded")
    
    print("✅ Model loading completed")
    return model, tokenizer

def extract_user_content_and_wrong(messages):
    """
    Extract user content from messages and wrong field
    System content is now fixed and defined globally
    
    Args:
        messages: List of messages
    
    Returns:
        user_content: User content
        wrong: Value of wrong field (boolean)
    """
    user_content = None
    wrong = False
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            user_content = content
        # wrong field may exist as an independent object in messages, or identified by "wrong" key
        elif "wrong" in msg:
            # If wrong field exists, read its value
            wrong_value = msg.get("wrong")
            # Handle different wrong value formats
            if isinstance(wrong_value, bool):
                wrong = wrong_value
            elif isinstance(wrong_value, str):
                wrong = wrong_value.lower() in ["true", "wrong"]
            else:
                wrong = bool(wrong_value)
    
    return user_content, wrong

def format_messages(user_content):
    """
    Format messages, combine fixed system content and user content
    
    Args:
        user_content: User content
    
    Returns:
        messages: Formatted list of messages
    """
    messages = []
    
    # Always use fixed SYSTEM_CONTENT
    messages.append({"role": "system", "content": SYSTEM_CONTENT})
    
    if user_content:
        messages.append({"role": "user", "content": user_content})
    
    return messages

def clean_assistant_response(text):
    """
    Clean irregular prefixes in assistant_response
    Remove "assistant\n\n" or "assistant\n" prefixes
    
    Args:
        text: Original assistant_response text
    
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Remove leading "assistant\n\n" or "assistant\n"
    # Process "assistant\n\n" first
    if text.startswith("assistant\n\n"):
        text = text[len("assistant\n\n"):]
    # Then process "assistant\n"
    elif text.startswith("assistant\n"):
        text = text[len("assistant\n"):]
    
    # Remove leading whitespace characters
    text = text.lstrip()
    
    return text

def process_data(model, tokenizer, json_file_path, output_file=None, max_samples=None, batch_size=1):
    """
    Process data from JSON file and feed to model continuously
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        json_file_path: JSON file path
        output_file: Output file path (optional, if provided will save results)
        max_samples: Maximum number of samples to process (optional, for testing)
        batch_size: Batch size, default is 1 (process one by one)
    """
    print(f"📖 Reading file: {json_file_path}")
    
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Read successfully, total {len(data)} entries")
    
    if max_samples:
        data = data[:max_samples]
        print(f"⚠️  Limiting to first {max_samples} entries")
    
    print(f"📦 Batch size: {batch_size}")
    
    results = []
    
    # Process data in batches
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Batch progress"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        
        # Process current batch
        batch_results = process_batch(model, tokenizer, batch_data, start_idx)
        results.extend(batch_results)
        
        # Display progress after each batch
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"\n✅ Processed: {end_idx}/{len(data)} entries ({batch_idx + 1}/{num_batches} batches)")
    
    print(f"\n📊 Processing statistics:")
    print(f"   Total data: {len(data)}")
    print(f"   Successfully processed: {len(results)}")
    
    # If output file is specified, save results
    if output_file:
        print(f"\n💾 Saving results to: {output_file}")
        # Only save user_content and assistant_response
        simplified_results = []
        for result in results:
            print(result)
            simplified_result = {
                "user_content": result.get("user_content", ""),
                "assistant_response": result.get("assistant_response", ""),
                "wrong": result.get("wrong", False)
            }
            simplified_results.append(simplified_result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved (containing {len(simplified_results)} entries)")
    
    return results

def process_batch(model, tokenizer, batch_data, start_idx):
    """
    Process one batch of data
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        batch_data: Current batch data list
        start_idx: Starting index of current batch
    
    Returns:
        results: List of processing results
    """
    results = []
    
    # If batch_size=1, process one by one (maintain original logic)
    if len(batch_data) == 1:
        item = batch_data[0]
        idx = start_idx
        try:
            messages = item.get("messages", [])
            
            # Extract user content and wrong field (system content is now fixed)
            user_content, wrong = extract_user_content_and_wrong(messages)
            
            if not user_content:
                print(f"⚠️  Entry {idx+1} has no user content, skipping")
                return []
            
            # Format messages (using fixed SYSTEM_CONTENT)
            messages = format_messages(user_content)
            
            if not messages:
                return []
            
            # Use tokenizer's chat template to format prompt
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Move input to model's device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract assistant's response (remove input part)
            input_ids = inputs["input_ids"][0]
            input_length = input_ids.shape[0]
            output_tokens = outputs[0][input_length:]
            assistant_response = tokenizer.decode(output_tokens, skip_special_tokens=True)
            
            # Clean irregular prefixes in assistant_response
            assistant_response = clean_assistant_response(assistant_response)
            
            result = {
                "index": idx,
                "system_content": SYSTEM_CONTENT,
                "user_content": user_content,
                "formatted_prompt": prompt,
                "assistant_response": assistant_response,
                "full_output": generated_text,
                "wrong": wrong
            }
            results.append(result)
        
        except Exception as e:
            print(f"\n❌ Error processing entry {idx+1}: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # batch_size > 1, batch processing
        batch_prompts = []
        batch_indices = []
        batch_user_contents = []
        batch_wrongs = []
        
        for item_idx, item in enumerate(batch_data):
            idx = start_idx + item_idx
            try:
                messages = item.get("messages", [])
                
                # Extract user content and wrong field (system content is now fixed)
                user_content, wrong = extract_user_content_and_wrong(messages)
                
                if not user_content:
                    print(f"⚠️  Entry {idx+1} has no user content, skipping")
                    continue
                
                # Format messages (using fixed SYSTEM_CONTENT)
                formatted_messages = format_messages(user_content)
                
                if not formatted_messages:
                    continue
                
                # Use tokenizer's chat template to format prompt
                prompt = tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                batch_prompts.append(prompt)
                batch_indices.append(idx)
                batch_user_contents.append(user_content)
                batch_wrongs.append(wrong)
            
            except Exception as e:
                print(f"\n❌ Error preparing entry {idx+1}: {e}")
                continue
        
        if not batch_prompts:
            return []
        
        try:
            # Batch tokenize
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096  # Adjust as needed
            )
            
            # Move input to model's device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Batch generate responses
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode each output
            for i, (idx, user_content, wrong, prompt) in enumerate(
                zip(batch_indices, batch_user_contents, batch_wrongs, batch_prompts)
            ):
                try:
                    # Extract assistant's response (remove input part)
                    input_ids = inputs["input_ids"][i]
                    # Calculate actual input length (excluding padding)
                    input_length = (input_ids != tokenizer.pad_token_id).sum().item()
                    # Extract newly generated tokens
                    output_tokens = outputs[i][input_length:]
                    assistant_response = tokenizer.decode(output_tokens, skip_special_tokens=True)
                    
                    # Clean irregular prefixes in assistant_response
                    assistant_response = clean_assistant_response(assistant_response)
                    
                    generated_text = tokenizer.decode(outputs[i], skip_special_tokens=False)
                    
                    result = {
                        "index": idx,
                        "system_content": SYSTEM_CONTENT,
                        "user_content": user_content,
                        "formatted_prompt": prompt,
                        "assistant_response": assistant_response,
                        "full_output": generated_text,
                        "wrong": wrong
                    }
                    results.append(result)
                
                except Exception as e:
                    print(f"\n❌ Error decoding entry {idx+1}: {e}")
                    continue
        
        except Exception as e:
            print(f"\n❌ Error in batch processing: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def main():
    """Main function"""
    # File paths (default values, can be overridden by command line arguments)
    json_file_path = "/root/autodl-tmp/LLaMA-Factory-main/data/prompt_for_gpt.json"
    # Default to using LoRA fine-tuned model
    model_path = "/root/autodl-tmp/LLaMA-Factory-main/saves/model/lora/sft"
    lora_path = None  # If model_path is a LoRA directory, will be auto-detected
    
    # Get paths from command line arguments (optional)
    # Argument order: json_file [model_path] [output_file] [max_samples] [batch_size]
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    # Output file (optional)
    output_file = "/root/autodl-tmp/LLaMA-Factory-main/data/llama_generate_new.json"
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    # Maximum number of samples (optional, for testing)
    max_samples = None
    if len(sys.argv) > 4:
        max_samples = int(sys.argv[4])
    
    # Batch size (optional, default is 1)
    # Reduced to 4 due to very long SYSTEM_CONTENT with 10 examples
    batch_size = 8
    if len(sys.argv) > 5:
        batch_size = int(sys.argv[5])
    
    # Check if file exists
    if not Path(json_file_path).exists():
        print(f"❌ Error: JSON file does not exist: {json_file_path}")
        return
    
    # Check model path (may be base model or LoRA adapter)
    if not Path(model_path).exists():
        print(f"❌ Error: Model path does not exist: {model_path}")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, lora_path)
    
    # Process data
    results = process_data(
        model=model,
        tokenizer=tokenizer,
        json_file_path=json_file_path,
        output_file=output_file,
        max_samples=max_samples,
        batch_size=batch_size
    )
    
    print("\n✅ Processing completed!")

if __name__ == "__main__":
    main()

