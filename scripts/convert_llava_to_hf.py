"""
Convert LLaVA custom checkpoint (LlavaQwenForCausalLM) to HuggingFace
LlavaOnevisionForConditionalGeneration format.

Key mapping:
  Checkpoint                                    -> HF LlavaOnevision
  model.layers.*                                -> model.language_model.layers.*
  model.embed_tokens.*                          -> model.language_model.embed_tokens.*
  model.norm.*                                  -> model.language_model.norm.*
  model.vision_tower.vision_tower.vision_model.* -> model.vision_tower.vision_model.*
  model.mm_projector.0.*                        -> model.multi_modal_projector.linear_1.*
  model.mm_projector.2.*                        -> model.multi_modal_projector.linear_2.*
  model.image_newline                           -> model.image_newline  (unchanged)
  lm_head.weight                                -> lm_head.weight      (unchanged)
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def remap_key(key: str) -> str:
    """Remap a single weight key from LLaVA custom format to HF format."""
    # Vision tower: remove extra .vision_tower level
    if key.startswith("model.vision_tower.vision_tower."):
        return key.replace("model.vision_tower.vision_tower.", "model.vision_tower.", 1)

    # Projector: mm_projector.0 -> multi_modal_projector.linear_1
    if key.startswith("model.mm_projector.0."):
        return key.replace("model.mm_projector.0.", "model.multi_modal_projector.linear_1.", 1)
    if key.startswith("model.mm_projector.2."):
        return key.replace("model.mm_projector.2.", "model.multi_modal_projector.linear_2.", 1)

    # Language model: model.layers.* -> model.language_model.layers.*
    if key.startswith("model.layers."):
        return key.replace("model.layers.", "model.language_model.layers.", 1)

    # Embedding: model.embed_tokens -> model.language_model.embed_tokens
    if key.startswith("model.embed_tokens."):
        return key.replace("model.embed_tokens.", "model.language_model.embed_tokens.", 1)

    # Norm: model.norm -> model.language_model.norm
    if key.startswith("model.norm."):
        return key.replace("model.norm.", "model.language_model.norm.", 1)

    # image_newline and lm_head stay the same
    return key


def convert_config(src_config: dict) -> dict:
    """Convert LLaVA custom config to HF LlavaOnevision config format."""
    # Read vision config from the vision tower's config
    vision_tower_name = src_config.get("mm_vision_tower", src_config.get("vision_tower", ""))

    hf_config = {
        "architectures": ["LlavaOnevisionForConditionalGeneration"],
        "model_type": "llava_onevision",
        "image_token_index": src_config.get("image_token_index", 151646),
        "video_token_index": src_config.get("video_token_index", 151647),
        "vision_feature_layer": src_config.get("mm_vision_select_layer", -2),
        "vision_feature_select_strategy": src_config.get("mm_vision_select_feature", "default"),
        # Text config (Qwen2)
        "text_config": {
            "model_type": "qwen2",
            "hidden_size": src_config.get("hidden_size", 3584),
            "intermediate_size": src_config.get("intermediate_size", 18944),
            "num_hidden_layers": src_config.get("num_hidden_layers", 28),
            "num_attention_heads": src_config.get("num_attention_heads", 28),
            "num_key_value_heads": src_config.get("num_key_value_heads", 4),
            "max_position_embeddings": src_config.get("max_position_embeddings", 32768),
            "vocab_size": src_config.get("vocab_size", 152064),
            "rms_norm_eps": src_config.get("rms_norm_eps", 1e-06),
            "rope_theta": src_config.get("rope_theta", 1000000.0),
            "hidden_act": src_config.get("hidden_act", "silu"),
            "attention_dropout": 0.0,
            "sliding_window": src_config.get("sliding_window", 32768),
            "use_sliding_window": src_config.get("use_sliding_window", False),
        },
        # Vision config (SigLIP)
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 26,
            "num_attention_heads": 16,
            "image_size": 384,
            "patch_size": 14,
        },
        # Projector
        "projector_hidden_act": "gelu",
        "torch_dtype": src_config.get("torch_dtype", "float16"),
    }

    return hf_config


def convert_weights(src_dir: str, dst_dir: str):
    """Convert weights from LLaVA custom format to HF format."""
    os.makedirs(dst_dir, exist_ok=True)

    # Load all safetensor files
    safetensor_files = sorted(
        [f for f in os.listdir(src_dir) if f.endswith(".safetensors")]
    )

    all_tensors = {}
    for sf in safetensor_files:
        print(f"  Loading {sf}...")
        with safe_open(os.path.join(src_dir, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    print(f"  Total source keys: {len(all_tensors)}")

    # Remap keys
    remapped = {}
    for old_key, tensor in all_tensors.items():
        new_key = remap_key(old_key)
        remapped[new_key] = tensor
        if old_key != new_key:
            pass  # silently remap

    print(f"  Total remapped keys: {len(remapped)}")

    # Save as single safetensors file (or sharded if large)
    total_size = sum(t.numel() * t.element_size() for t in remapped.values())
    print(f"  Total size: {total_size / 1e9:.2f} GB")

    # Shard into ~5GB files
    shard_size = 5 * 1024 * 1024 * 1024  # 5GB
    shards = []
    current_shard = {}
    current_size = 0
    weight_map = {}

    for key in sorted(remapped.keys()):
        tensor = remapped[key]
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    # Save shards
    index = {"metadata": {"total_size": total_size}, "weight_map": {}}
    for i, shard in enumerate(shards):
        if len(shards) == 1:
            fname = "model.safetensors"
        else:
            fname = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"

        print(f"  Saving {fname} ({len(shard)} tensors)...")
        save_file(shard, os.path.join(dst_dir, fname))

        for key in shard:
            index["weight_map"][key] = fname

    if len(shards) > 1:
        with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    return len(remapped)


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLaVA custom checkpoint to HF LlavaOnevision format"
    )
    parser.add_argument("--src", required=True, help="Source model directory")
    parser.add_argument("--dst", required=True, help="Destination directory")
    args = parser.parse_args()

    print(f"Converting {args.src} -> {args.dst}")

    # Convert config
    print("Converting config...")
    with open(os.path.join(args.src, "config.json")) as f:
        src_config = json.load(f)

    hf_config = convert_config(src_config)
    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    print("  Config saved.")

    # Convert weights
    print("Converting weights...")
    n_keys = convert_weights(args.src, args.dst)
    print(f"  Converted {n_keys} weight keys.")

    # Copy tokenizer files
    print("Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json",
        "merges.txt", "vocab.json",
    ]
    for fname in tokenizer_files:
        src = os.path.join(args.src, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.dst, fname))
            print(f"  Copied {fname}")

    # Copy preprocessor config
    for fname in ["preprocessor_config.json", "chat_template.json"]:
        src = os.path.join(args.src, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.dst, fname))
            print(f"  Copied {fname}")

    # Create generation_config.json
    gen_config = {
        "do_sample": False,
        "max_new_tokens": 512,
    }
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    print("\nConversion complete!")
    print(f"HF model saved to: {args.dst}")


if __name__ == "__main__":
    main()
