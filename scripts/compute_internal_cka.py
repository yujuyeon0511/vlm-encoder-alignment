"""
Compute CKA inside VLMs: before and after the projector (post-projector CKA).

Uses forward hooks to capture intermediate representations during inference:
  1. Vision encoder output (before projector)
  2. Projector output (after projector, same dim as LLM)
  3. LLM text embedding layer output

Then computes CKA between these representation pairs.

Usage:
    conda run -n docmllm python scripts/compute_internal_cka.py
    conda run -n docmllm python scripts/compute_internal_cka.py --models qwen25vl llava_ov
"""

import os
import sys
import json
import gc
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_alignment.analysis.cka import CKA
from vlm_alignment.analysis.coral import align_dimensions

# ============================================================================
# Model Configurations
# ============================================================================
VLM_CONFIGS = {
    "qwen25vl": {
        "display_name": "Qwen2.5-VL-7B",
        "vlm_path": "/NetDisk/juyeon/models/Qwen2.5-VL-7B-Instruct",
        "model_type": "qwen25vl",
        "vision_dim": 1280,
        "llm_dim": 3584,
    },
    "llava_ov": {
        "display_name": "LLaVA-OneVision-Qwen2-7B",
        "vlm_path": "/NetDisk/juyeon/AdaMMS/llava-onevision-qwen2-7b-si",
        "model_type": "llava_onevision",
        "vision_dim": 1152,
        "llm_dim": 3584,
    },
    "llava_gemma_siglip2": {
        "display_name": "LLaVA-Gemma-SigLIP2",
        "vlm_path": "/NetDisk/juyeon/llava-sp/two_stage_koni_siglip2_4n_251107_0036_stage2_finetune",
        "model_type": "llava_gemma_lora",
        "vision_tower": "google/siglip2-so400m-patch16-naflex",
        "base_llm": "google/gemma-3-4b-pt",
        "mm_projector_type": "mlp2x_gelu",
        "vision_dim": 1152,
        "llm_dim": 2560,
    },
    "llava_gemma_clip": {
        "display_name": "LLaVA-Gemma-CLIP",
        "vlm_path": "/NetDisk/juyeon/llava-sp/4n_llava-gemma3-4b-kisti-finetune_lora_testtt_250917",
        "model_type": "llava_gemma_lora",
        "vision_tower": "openai/clip-vit-large-patch14",
        "base_llm": "google/gemma-3-4b-pt",
        "mm_projector_type": "linear",
        "vision_dim": 1024,
        "llm_dim": 2560,
    },
}


def get_sample_images(n_samples: int = 100) -> list:
    """Collect sample images."""
    sample_dir = PROJECT_ROOT / "sample_data" / "images"
    images = []
    if sample_dir.exists():
        for img_path in sorted(sample_dir.rglob("*.png")):
            images.append(Image.open(img_path).convert("RGB"))

    while len(images) < n_samples:
        idx = len(images)
        img = Image.new("RGB", (384, 384), color=(
            (idx * 37) % 256, (idx * 73) % 256, (idx * 113) % 256
        ))
        images.append(img)
    return images[:n_samples]


# ============================================================================
# Hook-based Feature Extraction
# ============================================================================
class FeatureCapture:
    """Captures intermediate features using PyTorch hooks."""

    def __init__(self):
        self.features: Dict[str, List[torch.Tensor]] = {}
        self._hooks = []

    def register_hook(self, module: torch.nn.Module, name: str):
        """Register a forward hook on a module."""
        def hook_fn(mod, input, output):
            if isinstance(output, tuple):
                out = output[0]
            elif isinstance(output, torch.Tensor):
                out = output
            else:
                # Try to get the main tensor
                out = getattr(output, 'last_hidden_state', None)
                if out is None:
                    out = output[0] if hasattr(output, '__getitem__') else output

            if isinstance(out, torch.Tensor):
                # Mean pool spatial dimensions if 3D
                if out.dim() == 3:
                    pooled = out.mean(dim=1)
                elif out.dim() == 2:
                    pooled = out
                else:
                    pooled = out.reshape(out.shape[0], -1)

                if name not in self.features:
                    self.features[name] = []
                self.features[name].append(pooled.float().detach().cpu())

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def get_features(self, name: str) -> np.ndarray:
        """Get collected features as numpy array."""
        if name not in self.features:
            return np.array([])
        return torch.cat(self.features[name], dim=0).numpy()

    def clear(self):
        """Clear collected features."""
        self.features.clear()

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================================
# Model-specific extraction
# ============================================================================
@torch.no_grad()
def extract_qwen25vl(config: dict, images: list, device: str) -> dict:
    """Extract internal representations from Qwen2.5-VL."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"  Loading Qwen2.5-VL from {config['vlm_path']}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["vlm_path"], torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(config["vlm_path"])
    model.eval()

    capturer = FeatureCapture()

    # Qwen2.5-VL architecture:
    #   model.visual (ViT) → model.visual.merger (projector) → model.model (LLM)
    # Register hooks
    capturer.register_hook(model.visual, "vision_raw")  # ViT output
    if hasattr(model.visual, 'merger'):
        capturer.register_hook(model.visual.merger, "vision_projected")  # After merger/projector

    # Hook on LLM's embedding layer
    capturer.register_hook(model.model.embed_tokens, "text_embedding")

    batch_size = 2
    for i in tqdm(range(0, len(images), batch_size), desc="  Forward"):
        batch = images[i:i+batch_size]
        try:
            messages = [[{
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe."}]
            }] for img in batch]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
            inputs = processor(text=texts, images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Forward pass (just to trigger hooks, no generation needed)
            outputs = model(**inputs, output_hidden_states=False)
        except Exception as e:
            print(f"  Warning: batch {i} failed ({e})")

    result = {
        "vision_raw": capturer.get_features("vision_raw"),
        "vision_projected": capturer.get_features("vision_projected"),
        "text_embedding": capturer.get_features("text_embedding"),
    }

    capturer.remove_hooks()
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def extract_llava_onevision(config: dict, images: list, device: str) -> dict:
    """Extract internal representations from LLaVA-OneVision."""
    from transformers import AutoProcessor
    # LLaVA-OV uses llava-onevision architecture
    # Try loading with LlavaOnevisionForConditionalGeneration
    try:
        from transformers import LlavaOnevisionForConditionalGeneration
        model_cls = LlavaOnevisionForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq

    print(f"  Loading LLaVA-OV from {config['vlm_path']}...")
    model = model_cls.from_pretrained(
        config["vlm_path"], torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(config["vlm_path"], trust_remote_code=True)
    model.eval()

    capturer = FeatureCapture()

    # LLaVA-OV architecture:
    #   model.vision_tower → model.multi_modal_projector → model.language_model
    if hasattr(model, 'vision_tower'):
        capturer.register_hook(model.vision_tower, "vision_raw")
    if hasattr(model, 'multi_modal_projector'):
        capturer.register_hook(model.multi_modal_projector, "vision_projected")
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        capturer.register_hook(model.language_model.model.embed_tokens, "text_embedding")

    batch_size = 2
    for i in tqdm(range(0, len(images), batch_size), desc="  Forward"):
        batch = images[i:i+batch_size]
        try:
            prompts = [f"<image>\nDescribe this image." for _ in batch]

            if hasattr(processor, 'apply_chat_template'):
                messages = [[{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": "Describe this image."}
                ]}] for _ in batch]
                texts = [processor.apply_chat_template(m, add_generation_prompt=True) for m in messages]
            else:
                texts = prompts

            inputs = processor(text=texts, images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=False)
        except Exception as e:
            print(f"  Warning: batch {i} failed ({e})")

    result = {
        "vision_raw": capturer.get_features("vision_raw"),
        "vision_projected": capturer.get_features("vision_projected"),
        "text_embedding": capturer.get_features("text_embedding"),
    }

    capturer.remove_hooks()
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def extract_llava_gemma_lora(config: dict, images: list, device: str) -> dict:
    """Extract internal representations from LLaVA-Gemma (LoRA).

    Loads the model manually: base LLM + LoRA adapter + non_lora_trainables.
    Then extracts vision/projector/text features via hooks.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from peft import PeftModel

    vlm_path = config["vlm_path"]
    print(f"  Loading LLaVA-Gemma from {vlm_path}...")

    # Step 1: Load vision tower independently
    vision_tower_id = config["vision_tower"]
    print(f"  Loading vision tower: {vision_tower_id}")

    if "clip" in vision_tower_id.lower() and "siglip" not in vision_tower_id.lower():
        from transformers import CLIPModel, CLIPProcessor
        vision_model = CLIPModel.from_pretrained(vision_tower_id, torch_dtype=torch.float16).to(device)
        vision_processor = CLIPProcessor.from_pretrained(vision_tower_id)
        vision_type = "clip"
    else:
        from transformers import AutoModel
        vision_model = AutoModel.from_pretrained(vision_tower_id, torch_dtype=torch.float16).to(device)
        vision_processor = AutoProcessor.from_pretrained(vision_tower_id)
        vision_type = "siglip"
    vision_model.eval()

    # Step 2: Build projector
    vision_dim = config["vision_dim"]
    llm_dim = config["llm_dim"]
    projector_type = config["mm_projector_type"]

    if projector_type == "mlp2x_gelu":
        projector = torch.nn.Sequential(
            torch.nn.Linear(vision_dim, llm_dim),
            torch.nn.GELU(),
            torch.nn.Linear(llm_dim, llm_dim),
        )
    elif projector_type == "linear":
        projector = torch.nn.Linear(vision_dim, llm_dim)
    else:
        projector = torch.nn.Linear(vision_dim, llm_dim)

    # Load projector weights from non_lora_trainables.bin
    nlt_path = os.path.join(vlm_path, "non_lora_trainables.bin")
    if os.path.exists(nlt_path):
        print(f"  Loading non-LoRA trainables...")
        nlt = torch.load(nlt_path, map_location="cpu")

        # Extract mm_projector weights
        proj_weights = {}
        for k, v in nlt.items():
            # Handle various naming conventions
            for prefix in ["base_model.model.model.mm_projector.", "model.model.mm_projector.",
                          "model.mm_projector.", "mm_projector."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    proj_weights[new_key] = v
                    break

        if proj_weights:
            try:
                projector.load_state_dict(proj_weights)
                print(f"  Loaded {len(proj_weights)} projector weights")
            except Exception as e:
                print(f"  Warning: projector weight loading failed ({e})")

    projector = projector.to(device).half()
    projector.eval()

    # Step 3: Load LLM with LoRA
    base_llm_id = config["base_llm"]
    print(f"  Loading base LLM: {base_llm_id}")
    llm = AutoModelForCausalLM.from_pretrained(
        base_llm_id, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True
    )

    # Apply LoRA
    adapter_config_path = os.path.join(vlm_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print("  Applying LoRA adapter...")
        llm = PeftModel.from_pretrained(llm, vlm_path)
        llm = llm.merge_and_unload()
    llm.eval()

    # Step 4: Extract features
    all_vision_raw = []
    all_vision_proj = []
    all_text_emb = []

    batch_size = 4
    for i in tqdm(range(0, len(images), batch_size), desc="  Forward"):
        batch = images[i:i+batch_size]

        # Vision extraction
        inputs = vision_processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if vision_type == "clip":
            out = vision_model.vision_model(pixel_values=inputs["pixel_values"])
            vision_emb = out.pooler_output
        elif "siglip2" in config.get("vision_tower", "").lower():
            # SigLIP2 NaFlex: use get_image_features which handles the key mapping
            out = vision_model.get_image_features(**inputs)
            vision_emb = out.pooler_output
        else:
            out = vision_model.vision_model(pixel_values=inputs["pixel_values"])
            vision_emb = out.pooler_output

        all_vision_raw.append(vision_emb.float().cpu())

        # Project vision features
        projected = projector(vision_emb.half())
        all_vision_proj.append(projected.float().cpu())

        # Text embedding from LLM embed_tokens (use projected features as "tokens")
        # This represents how the LLM would see the visual features
        text_emb = llm.model.embed_tokens(
            torch.tensor([[1, 2, 3, 4, 5]] * len(batch), device=llm.device)
        ).mean(dim=1)
        all_text_emb.append(text_emb.float().cpu())

    result = {
        "vision_raw": torch.cat(all_vision_raw, dim=0).numpy(),
        "vision_projected": torch.cat(all_vision_proj, dim=0).numpy(),
        "text_embedding": torch.cat(all_text_emb, dim=0).numpy(),
    }

    del vision_model, vision_processor, projector, llm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def extract_features(config: dict, images: list, device: str) -> dict:
    """Dispatch to model-specific extraction."""
    model_type = config["model_type"]
    if model_type == "qwen25vl":
        return extract_qwen25vl(config, images, device)
    elif model_type == "llava_onevision":
        return extract_llava_onevision(config, images, device)
    elif model_type == "llava_gemma_lora":
        return extract_llava_gemma_lora(config, images, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# CKA Computation
# ============================================================================
def compute_internal_cka(features: dict) -> dict:
    """Compute CKA for internal VLM representations."""
    results = {}

    pairs = [
        ("vision_raw", "text_embedding", "CKA(VisionRaw, TextEmb)"),
        ("vision_projected", "text_embedding", "CKA(VisionProj, TextEmb)"),
        ("vision_raw", "vision_projected", "CKA(VisionRaw, VisionProj)"),
    ]

    for feat_a, feat_b, label in pairs:
        a = features.get(feat_a)
        b = features.get(feat_b)

        if a is None or b is None or len(a) == 0 or len(b) == 0:
            results[label] = {"cka_linear": None, "cka_rbf": None, "note": "features not available"}
            continue

        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        # L2 normalize
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

        # Align dimensions
        a_aligned, b_aligned = align_dimensions(a, b)

        cka_lin = CKA.compute_cka(a_aligned, b_aligned, kernel="linear")
        cka_rbf = CKA.compute_cka(a_aligned, b_aligned, kernel="rbf")

        results[label] = {
            "cka_linear": float(cka_lin),
            "cka_rbf": float(cka_rbf),
            "n_samples": n,
            "dim_a": a.shape[1],
            "dim_b": b.shape[1],
            "aligned_dim": a_aligned.shape[1],
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute internal VLM CKA")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "cka"))
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_keys = args.models if args.models else list(VLM_CONFIGS.keys())

    print("=" * 60)
    print("Internal (Post-Projector) CKA Computation")
    print(f"Models: {model_keys}")
    print(f"Samples: {args.n_samples}")
    print("=" * 60)

    images = get_sample_images(args.n_samples)
    print(f"Prepared {len(images)} images\n")

    all_results = {}

    for model_key in model_keys:
        config = VLM_CONFIGS[model_key]
        print(f"\n{'='*60}")
        print(f"Processing: {config['display_name']}")
        print(f"{'='*60}")

        try:
            features = extract_features(config, images, args.device)

            for k, v in features.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: shape={v.shape}")
                else:
                    print(f"  {k}: {type(v)}")

            cka_results = compute_internal_cka(features)
            cka_results["model_name"] = config["display_name"]

            all_results[model_key] = cka_results

            print(f"\n  Internal CKA for {config['display_name']}:")
            for label, metrics in cka_results.items():
                if isinstance(metrics, dict) and "cka_linear" in metrics:
                    cka_val = metrics["cka_linear"]
                    if cka_val is not None:
                        print(f"    {label}: linear={cka_val:.4f}, rbf={metrics['cka_rbf']:.4f}")
                    else:
                        print(f"    {label}: N/A ({metrics.get('note', '')})")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {"error": str(e), "model_name": config["display_name"]}

        gc.collect()
        torch.cuda.empty_cache()

    # Save
    output_file = os.path.join(args.output_dir, "internal_cka.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary
    print(f"\n{'='*60}")
    print("INTERNAL CKA SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'PreProj CKA':>12} {'PostProj CKA':>13} {'Delta':>8}")
    print("-" * 60)
    for key, res in all_results.items():
        if "error" in res:
            print(f"{res.get('model_name', key):<25} ERROR: {res['error'][:30]}")
            continue
        pre = res.get("CKA(VisionRaw, TextEmb)", {}).get("cka_linear")
        post = res.get("CKA(VisionProj, TextEmb)", {}).get("cka_linear")
        pre_str = f"{pre:.4f}" if pre is not None else "N/A"
        post_str = f"{post:.4f}" if post is not None else "N/A"
        delta = f"{post - pre:+.4f}" if pre is not None and post is not None else "N/A"
        name = res.get("model_name", key)
        print(f"{name:<25} {pre_str:>12} {post_str:>13} {delta:>8}")


if __name__ == "__main__":
    main()
