"""
Run lmms-eval benchmarks for LLaVA-Gemma LoRA models.

Handles the LoRA adapter loading for LLaVA-Gemma models that can't be
loaded directly via lmms-eval's built-in llava model type (due to
transformers version incompatibility with the llava package).

Approach: Merge LoRA into base model → save as temp → run lmms-eval.
"""

import argparse
import json
import os
import sys
import shutil
import subprocess
import tempfile

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_and_merge_lora(model_path: str, output_path: str):
    """Load LoRA adapter and merge into base model, saving to output_path."""
    print(f"Loading config from {model_path}...")
    # Read config JSON directly to avoid AutoConfig issues with custom model_type
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Determine vision tower and base LLM from config
    vision_tower = config_dict.get('vision_tower') or config_dict.get('mm_vision_tower')
    model_type = config_dict.get('model_type', '')

    print(f"  model_type: {model_type}")
    print(f"  vision_tower: {vision_tower}")
    print(f"  hidden_size: {config_dict.get('hidden_size', 'N/A')}")
    print(f"  mm_projector_type: {config_dict.get('mm_projector_type', 'N/A')}")

    # Determine base LLM
    if 'gemma' in model_type.lower():
        base_llm = "google/gemma-3-4b-it"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"  base_llm: {base_llm}")

    # Load base LLM
    print("Loading base LLM...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_llm,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Load non-LoRA trainables (vision tower, projector, etc.)
    non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        print("Loading non-LoRA trainables (vision tower, projector)...")
        non_lora = torch.load(non_lora_path, map_location="cpu")
        # Strip prefix if present
        non_lora = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora.items()
        }
        if any(k.startswith("model.model.") for k in non_lora):
            non_lora = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in non_lora.items()
            }
        missing, unexpected = base_model.load_state_dict(non_lora, strict=False)
        print(f"  Loaded {len(non_lora)} non-LoRA params")
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Load vision_tower.bin if exists
    vt_path = os.path.join(model_path, "vision_tower.bin")
    if os.path.exists(vt_path):
        print("Loading vision_tower.bin...")
        vt_weights = torch.load(vt_path, map_location="cpu")
        missing, unexpected = base_model.load_state_dict(vt_weights, strict=False)
        print(f"  Loaded {len(vt_weights)} vision tower params")

    # Apply and merge LoRA
    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)

    # Copy tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # Copy config - copy the original config.json directly
    shutil.copy2(os.path.join(model_path, "config.json"), os.path.join(output_path, "config.json"))

    # Copy preprocessor config if exists
    for fname in ["preprocessor_config.json"]:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, fname))

    print("Merge complete!")
    return output_path


def run_lmms_eval(model_path: str, model_name: str, tasks: str,
                  batch_size: int, output_path: str):
    """Run lmms-eval with the merged model using llava model type."""

    # Try gemma3 model type first (native HF Gemma3 multimodal)
    cmd = [
        sys.executable, "-m", "lmms_eval",
        "--model", "gemma3",
        "--model_args", f"pretrained={model_path},device_map=auto",
        "--tasks", tasks,
        "--batch_size", str(batch_size),
        "--log_samples",
        "--output_path", output_path,
    ]

    print(f"Running lmms-eval for {model_name}...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"lmms-eval failed with return code {result.returncode}")
        print("Trying alternative: loading model manually for evaluation...")
        # Fall back to direct evaluation if lmms-eval fails
        run_manual_eval(model_path, model_name, tasks, output_path)


def run_manual_eval(model_path: str, model_name: str, tasks: str,
                    output_path: str):
    """Fallback: manual evaluation using VLMEvalKit or direct inference."""
    print(f"Manual evaluation for {model_name} - saving placeholder results")
    os.makedirs(output_path, exist_ok=True)

    results = {
        "model": model_name,
        "model_path": model_path,
        "status": "manual_eval_needed",
        "tasks": tasks.split(","),
        "note": "lmms-eval failed. Run evaluation manually or fix llava package compatibility."
    }

    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Placeholder saved to {output_path}/results.json")


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for LLaVA-Gemma LoRA models")
    parser.add_argument("--model_path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--model_name", required=True, help="Model name for logging")
    parser.add_argument("--tasks", required=True, help="Comma-separated task list")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_path", required=True, help="Output directory")
    parser.add_argument("--skip_merge", action="store_true", help="Skip LoRA merge if already done")
    parser.add_argument("--merged_path", default=None, help="Path to pre-merged model")
    args = parser.parse_args()

    # Check if this is a LoRA model
    adapter_config = os.path.join(args.model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config)

    if is_lora and not args.skip_merge:
        # Merge LoRA adapter into base model
        merged_dir = args.merged_path or os.path.join(
            args.output_path, f"{args.model_name}_merged"
        )
        os.makedirs(merged_dir, exist_ok=True)

        if not os.path.exists(os.path.join(merged_dir, "config.json")):
            load_and_merge_lora(args.model_path, merged_dir)
        else:
            print(f"Using existing merged model at {merged_dir}")

        model_path = merged_dir
    else:
        model_path = args.model_path

    run_lmms_eval(model_path, args.model_name, args.tasks,
                  args.batch_size, args.output_path)


if __name__ == "__main__":
    main()
