# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging as llamafactory_logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.ktransformers import load_kt_pretrained_model
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = llamafactory_logging.get_logger(__name__)


class ModelNameFilter(logging.Filter):
    """Filter to replace llama-related names with generic 'model' in log messages."""
    
    def filter(self, record):
        # Process msg attribute
        if hasattr(record, 'msg') and record.msg:
            msg = str(record.msg)
            msg = self._replace_names(msg)
            record.msg = msg
        
        # Process formatted message (getMessage())
        if hasattr(record, 'getMessage'):
            try:
                formatted_msg = record.getMessage()
                if formatted_msg:
                    formatted_msg = self._replace_names(formatted_msg)
                    # Update the message in the record
                    record.msg = formatted_msg
            except Exception:
                pass
        
        return True
    
    def _replace_names(self, text):
        """Replace llama-related names with generic 'model'."""
        # Replace LlamaConfig with ModelConfig
        text = text.replace("LlamaConfig", "ModelConfig")
        # Replace LlamaForCausalLM with ModelForCausalLM
        text = text.replace("LlamaForCausalLM", "ModelForCausalLM")
        # Replace model_type: "llama" with model_type: "model"
        text = text.replace('"model_type": "llama"', '"model_type": "model"')
        # Replace rope_type: "llama3" with rope_type: "model"
        text = text.replace('"rope_type": "llama3"', '"rope_type": "model"')
        text = text.replace('"rope_type": "llama', '"rope_type": "model')
        return text


# Create and add filter for transformers loggers
_model_name_filter = ModelNameFilter()
# Apply to configuration_utils logger (where config loading logs come from)
_transformers_config_logger = logging.getLogger("transformers.configuration_utils")
_transformers_config_logger.addFilter(_model_name_filter)
# Also apply to other transformers loggers that might print config info
_transformers_modeling_logger = logging.getLogger("transformers.modeling_utils")
_transformers_modeling_logger.addFilter(_model_name_filter)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    import json
    import os as os_module
    
    init_kwargs = _get_init_kwargs(model_args)
    config_path = os.path.join(model_args.model_name_or_path, "config.json")
    
    # Use process ID to avoid conflicts in multi-process environment
    pid = os_module.getpid()
    original_config_path = config_path + f".orig_llamafactory_tokenizer_{pid}"
    config_modified = False
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Always use fixed values "llama" and "llama3" for loading
            if config_dict.get('model_type') != 'llama':
                config_dict['model_type'] = 'llama'
                config_modified = True
            
            if 'rope_scaling' in config_dict and isinstance(config_dict['rope_scaling'], dict):
                if config_dict['rope_scaling'].get('rope_type') != 'llama3':
                    config_dict['rope_scaling']['rope_type'] = 'llama3'
                    config_modified = True
            
            if config_modified:
                # Save original config with process-specific name
                if os.path.exists(original_config_path):
                    os.remove(original_config_path)
                os.rename(config_path, original_config_path)
                
                # Write modified config
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=model_args.use_fast_tokenizer,
                split_special_tokens=model_args.split_special_tokens,
                padding_side="right",
                **init_kwargs,
            )
        except ValueError:  # try another one
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=not model_args.use_fast_tokenizer,
                padding_side="right",
                **init_kwargs,
            )
    except Exception as e:
        # Restore config before raising
        if config_modified and os.path.exists(original_config_path):
            if os.path.exists(config_path):
                os.remove(config_path)
            os.rename(original_config_path, config_path)
        raise OSError("Failed to load tokenizer.") from e
    finally:
        # Restore original config.json after tokenizer loading
        if config_modified and os.path.exists(original_config_path):
            if os.path.exists(config_path):
                os.remove(config_path)
            os.rename(original_config_path, config_path)

    patch_tokenizer(tokenizer, model_args)

    # Reload config for processor if needed
    config_modified_processor = False
    original_config_path_processor = config_path + f".orig_llamafactory_processor_{pid}"
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            if config_dict.get('model_type') != 'llama':
                config_dict['model_type'] = 'llama'
                config_modified_processor = True
            
            if 'rope_scaling' in config_dict and isinstance(config_dict['rope_scaling'], dict):
                if config_dict['rope_scaling'].get('rope_type') != 'llama3':
                    config_dict['rope_scaling']['rope_type'] = 'llama3'
                    config_modified_processor = True
            
            if config_modified_processor:
                if os.path.exists(original_config_path_processor):
                    os.remove(original_config_path_processor)
                os.rename(config_path, original_config_path_processor)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        try:
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                use_fast=model_args.use_fast_tokenizer,
                **init_kwargs,
            )
        except ValueError:  # try another one
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                use_fast=not model_args.use_fast_tokenizer,
                **init_kwargs,
            )
        except Exception as e:
            logger.info_rank0(f"Failed to load processor: {e}.")
            processor = None
    finally:
        # Restore original config.json after processor loading
        if config_modified_processor and os.path.exists(original_config_path_processor):
            if os.path.exists(config_path):
                os.remove(config_path)
            os.rename(original_config_path_processor, config_path)

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config.
    
    Ignores model_type and rope_type in config.json, always uses fixed values
    "llama" and "llama3" for loading, but displays as "model" in logs.
    """
    import json
    import os
    
    init_kwargs = _get_init_kwargs(model_args)
    config_path = os.path.join(model_args.model_name_or_path, "config.json")
    
    # Always use fixed values "llama" and "llama3" for loading, ignore config.json
    # Use process ID to avoid conflicts in multi-process environment
    import os as os_module
    pid = os_module.getpid()
    original_config_path = config_path + f".orig_llamafactory_{pid}"
    temp_config_path = config_path + f".temp_llamafactory_{pid}"
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
            
            # Temporarily write modified config for loading
            if config_modified:
                # Save original config with process-specific name
                if os.path.exists(original_config_path):
                    os.remove(original_config_path)
                os.rename(config_path, original_config_path)
                
                # Write modified config
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # Load config with fixed values
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        
    finally:
        # Restore original config.json
        if config_modified:
            if os.path.exists(original_config_path):
                if os.path.exists(config_path):
                    os.remove(config_path)
                os.rename(original_config_path, config_path)
    
    # Always use "llama" and "llama3" for functionality, display as "model"
    config._original_model_type = 'llama'
    config.model_type = 'model'  # Display value
    
    if hasattr(config, 'architectures') and config.architectures:
        config._original_architectures = config.architectures.copy()
        config.architectures = [arch.replace('Llama', 'Model') for arch in config.architectures]
    
    # Handle rope_type: always use "llama3" for functionality, display as "model"
    if hasattr(config, 'rope_scaling'):
        rope_scaling = config.rope_scaling
        if isinstance(rope_scaling, dict) and 'rope_type' in rope_scaling:
            rope_scaling['_original_rope_type'] = 'llama3'
            rope_scaling['rope_type'] = 'model'  # Display value
    
    # Patch __repr__ to show generic names (only for display, actual values unchanged)
    original_repr = config.__repr__
    def patched_repr(self):
        repr_str = original_repr()
        # Replace LlamaConfig with ModelConfig
        repr_str = repr_str.replace("LlamaConfig", "ModelConfig")
        # Replace llama with model in model_type
        repr_str = repr_str.replace('"model_type": "llama"', '"model_type": "model"')
        # Replace LlamaForCausalLM with ModelForCausalLM
        repr_str = repr_str.replace("LlamaForCausalLM", "ModelForCausalLM")
        # Replace llama3 with model in rope_type (handle both "llama3" and other llama variants)
        repr_str = repr_str.replace('"rope_type": "llama3"', '"rope_type": "model"')
        repr_str = repr_str.replace('"rope_type": "llama', '"rope_type": "model')
        return repr_str
    
    config.__repr__ = MethodType(patched_repr, config)
    
    return config


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_kt:
        from ktransformers.sft.monkey_patch_torch_module import install_patch

        install_patch()
        model = load_kt_pretrained_model(config, model_args)
    elif model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
                    model = getattr(model, "thinker")

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
