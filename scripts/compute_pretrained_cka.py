"""
Compute CKA between pretrained Vision Encoders and LLMs (before VLM training).

Measures the "raw" alignment between independent vision and language models
to establish a baseline before any projection/adaptation.

Usage:
    conda run -n docmllm python scripts/compute_pretrained_cka.py
    conda run -n docmllm python scripts/compute_pretrained_cka.py --n_samples 200
"""

import os
import sys
import json
import gc
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA

# Add project root to path
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
        "vision_encoder": "qwen25vl_vit",  # self-contained ViT, extract from VLM
        "vision_model_id": None,  # extracted from VLM
        "vlm_path": "/NetDisk/juyeon/models/Qwen2.5-VL-7B-Instruct",
        "llm_model_id": "Qwen/Qwen2.5-7B",
        "vision_dim": 1280,
        "llm_dim": 3584,
    },
    "llava_ov": {
        "display_name": "LLaVA-OneVision-Qwen2-7B",
        "vision_encoder": "siglip_so400m",
        "vision_model_id": "google/siglip-so400m-patch14-384",
        "vlm_path": "/NetDisk/juyeon/AdaMMS/llava-onevision-qwen2-7b-si",
        "llm_model_id": "Qwen/Qwen2-7B",
        "vision_dim": 1152,
        "llm_dim": 3584,
    },
    "llava_gemma_siglip2": {
        "display_name": "LLaVA-Gemma-SigLIP2",
        "vision_encoder": "siglip2_so400m",
        "vision_model_id": "google/siglip2-so400m-patch16-naflex",
        "vlm_path": "/NetDisk/juyeon/llava-sp/two_stage_koni_siglip2_4n_251107_0036_stage2_finetune",
        "llm_model_id": "google/gemma-3-4b-pt",
        "vision_dim": 1152,
        "llm_dim": 2560,
    },
    "llava_gemma_clip": {
        "display_name": "LLaVA-Gemma-CLIP",
        "vision_encoder": "clip_vit_l14",
        "vision_model_id": "openai/clip-vit-large-patch14",
        "vlm_path": "/NetDisk/juyeon/llava-sp/4n_llava-gemma3-4b-kisti-finetune_lora_testtt_250917",
        "llm_model_id": "google/gemma-3-4b-pt",
        "vision_dim": 1024,
        "llm_dim": 2560,
    },
}

# Text prompts for LLM embedding extraction (diverse queries for benchmark domains)
TEXT_PROMPTS = [
    "What is the value shown in this chart?",
    "Describe the data in this table.",
    "What information does this document contain?",
    "Read the text in the image.",
    "What are the main trends shown?",
    "Summarize the key findings.",
    "What is the title of this chart?",
    "Compare the values across categories.",
    "What type of visualization is this?",
    "Extract the numbers from this table.",
    "What is the conclusion drawn from this data?",
    "Identify the outliers in this chart.",
    "What labels appear on the axes?",
    "Describe the layout of this document.",
    "What is the relationship between the variables?",
    "List all text visible in the image.",
    "What pattern do you observe?",
    "How many rows and columns are there?",
    "What color scheme is used?",
    "Is there a legend in this figure?",
]


def get_sample_images(n_samples: int = 100) -> list:
    """Collect sample images from project sample_data or generate synthetic ones."""
    sample_dir = PROJECT_ROOT / "sample_data" / "images"
    images = []

    # Collect from sample_data
    if sample_dir.exists():
        for img_path in sorted(sample_dir.rglob("*.png")):
            images.append(Image.open(img_path).convert("RGB"))
    for img_path in sorted(sample_dir.rglob("*.jpg")) if sample_dir.exists() else []:
        images.append(Image.open(img_path).convert("RGB"))

    # If not enough, generate synthetic images
    while len(images) < n_samples:
        # Create simple synthetic images with varying content
        idx = len(images)
        img = Image.new("RGB", (384, 384), color=(
            (idx * 37) % 256,
            (idx * 73) % 256,
            (idx * 113) % 256,
        ))
        images.append(img)

    return images[:n_samples]


def get_text_prompts(n_samples: int = 100) -> list:
    """Generate text prompts for LLM embedding extraction."""
    prompts = []
    while len(prompts) < n_samples:
        prompts.extend(TEXT_PROMPTS)
    return prompts[:n_samples]


# ============================================================================
# Vision Encoder Extraction
# ============================================================================
@torch.no_grad()
def extract_vision_embeddings_siglip(model_id: str, images: list, device: str) -> np.ndarray:
    """Extract embeddings from SigLIP/SigLIP2 model."""
    from transformers import AutoModel, AutoProcessor
    print(f"  Loading SigLIP from {model_id}...")
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    is_siglip2 = "siglip2" in model_id.lower()

    all_embs = []
    batch_size = 8
    for i in tqdm(range(0, len(images), batch_size), desc="  Vision"):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if is_siglip2:
            # SigLIP2 NaFlex: processor returns pixel_attention_mask but
            # vision_model expects attention_mask. Use get_image_features instead.
            outputs = model.get_image_features(**inputs)
            embs = outputs.pooler_output
        else:
            outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            embs = outputs.pooler_output

        all_embs.append(embs.float().cpu().numpy())

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def extract_vision_embeddings_clip(model_id: str, images: list, device: str) -> np.ndarray:
    """Extract embeddings from CLIP model."""
    from transformers import CLIPModel, CLIPProcessor
    print(f"  Loading CLIP from {model_id}...")
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    all_embs = []
    batch_size = 8
    for i in tqdm(range(0, len(images), batch_size), desc="  Vision"):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.vision_model(pixel_values=inputs["pixel_values"])
        embs = out.pooler_output.float().cpu().numpy()
        all_embs.append(embs)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


@torch.no_grad()
def extract_vision_embeddings_qwen25vl(vlm_path: str, images: list, device: str) -> np.ndarray:
    """Extract vision embeddings from Qwen2.5-VL's built-in ViT."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print(f"  Loading Qwen2.5-VL ViT from {vlm_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(vlm_path)
    model.eval()

    # Extract vision encoder part: model.model.visual
    # Qwen2.5-VL concatenates all image patches into a single sequence (no batch dim)
    # Process one image at a time to get per-image embeddings
    vision_model = model.model.visual

    all_embs = []
    for i in tqdm(range(len(images)), desc="  Vision"):
        img = images[i]
        try:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe."}
            ]}]
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = processor(text=[text], images=[img], return_tensors="pt", padding=True)
            inp = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inp.items()}

            pixel_values = inp["pixel_values"]
            grid_thw = inp.get("image_grid_thw", None)
            if grid_thw is not None:
                vision_out = vision_model(pixel_values, grid_thw=grid_thw)
            else:
                vision_out = vision_model(pixel_values)

            # Qwen2.5-VL visual returns BaseModelOutputWithPooling
            # last_hidden_state shape: (total_tokens, hidden_dim) for single image
            if hasattr(vision_out, 'last_hidden_state'):
                hidden = vision_out.last_hidden_state
            elif isinstance(vision_out, torch.Tensor):
                hidden = vision_out
            else:
                hidden = vision_out[0]

            # Mean pool: (tokens, dim) -> (1, dim)
            if hidden.dim() == 2:
                emb = hidden.mean(dim=0, keepdim=True)
            elif hidden.dim() == 3:
                emb = hidden.mean(dim=1)
            else:
                emb = hidden.reshape(1, -1)

            all_embs.append(emb.float().cpu().numpy())
        except Exception as e:
            print(f"  Warning: image {i} failed ({e}), using zeros")
            all_embs.append(np.zeros((1, 1280), dtype=np.float32))

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    result = np.concatenate(all_embs, axis=0)
    # Ensure consistent number of samples
    return result[:len(images)]


def extract_vision_embeddings(config: dict, images: list, device: str) -> np.ndarray:
    """Dispatch to the right vision encoder extraction function."""
    encoder_type = config["vision_encoder"]

    if encoder_type == "qwen25vl_vit":
        return extract_vision_embeddings_qwen25vl(config["vlm_path"], images, device)
    elif "siglip" in encoder_type:
        return extract_vision_embeddings_siglip(config["vision_model_id"], images, device)
    elif "clip" in encoder_type:
        return extract_vision_embeddings_clip(config["vision_model_id"], images, device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ============================================================================
# LLM Text Embedding Extraction
# ============================================================================
@torch.no_grad()
def extract_llm_embeddings(model_id: str, texts: list, device: str) -> np.ndarray:
    """Extract text embeddings from a pretrained LLM."""
    from transformers import AutoModel, AutoTokenizer
    print(f"  Loading LLM from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embs = []
    batch_size = 8
    for i in tqdm(range(0, len(texts), batch_size), desc="  Text"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=128
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embs.append(pooled.float().cpu().numpy())

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


# ============================================================================
# CKA Computation
# ============================================================================
def compute_cka_metrics(vision_emb: np.ndarray, text_emb: np.ndarray) -> dict:
    """Compute CKA metrics between vision and text embeddings."""
    n = min(vision_emb.shape[0], text_emb.shape[0])
    v = vision_emb[:n]
    t = text_emb[:n]

    # L2 normalize
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)

    # Align dimensions via PCA if needed
    v_aligned, t_aligned = align_dimensions(v, t)

    # CKA computation
    cka_linear = CKA.compute_cka(v_aligned, t_aligned, kernel="linear")
    cka_rbf = CKA.compute_cka(v_aligned, t_aligned, kernel="rbf")

    # Also compute without PCA alignment (using raw dimensions)
    cka_linear_raw = CKA.compute_cka(v, t, kernel="linear")

    # Cosine similarity (after PCA alignment)
    cos_sim = np.mean(np.sum(v_aligned * t_aligned, axis=1))

    return {
        "cka_linear": float(cka_linear),
        "cka_rbf": float(cka_rbf),
        "cka_linear_raw": float(cka_linear_raw),
        "cosine_similarity": float(cos_sim),
        "n_samples": n,
        "vision_dim": vision_emb.shape[1],
        "text_dim": text_emb.shape[1],
        "aligned_dim": v_aligned.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute pretrained CKA")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples for CKA computation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "cka"),
                        help="Output directory")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to evaluate (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Select models
    model_keys = args.models if args.models else list(VLM_CONFIGS.keys())

    print("=" * 60)
    print("Pretrained CKA Computation")
    print(f"Models: {model_keys}")
    print(f"Samples: {args.n_samples}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Prepare data
    images = get_sample_images(args.n_samples)
    texts = get_text_prompts(args.n_samples)
    print(f"Prepared {len(images)} images and {len(texts)} texts\n")

    results = {}

    # Cache LLM embeddings to avoid recomputing for same LLM
    llm_cache = {}

    for model_key in model_keys:
        config = VLM_CONFIGS[model_key]
        print(f"\n{'='*60}")
        print(f"Processing: {config['display_name']}")
        print(f"  Vision: {config['vision_encoder']} ({config['vision_dim']}d)")
        print(f"  LLM: {config['llm_model_id']} ({config['llm_dim']}d)")
        print(f"{'='*60}")

        # Extract vision embeddings
        print("\n[1/3] Extracting vision embeddings...")
        vision_emb = extract_vision_embeddings(config, images, args.device)
        print(f"  Vision shape: {vision_emb.shape}")

        # Extract LLM embeddings (with caching)
        llm_id = config["llm_model_id"]
        if llm_id in llm_cache:
            print(f"\n[2/3] Using cached LLM embeddings for {llm_id}")
            text_emb = llm_cache[llm_id]
        else:
            print(f"\n[2/3] Extracting LLM embeddings from {llm_id}...")
            text_emb = extract_llm_embeddings(llm_id, texts, args.device)
            llm_cache[llm_id] = text_emb
        print(f"  Text shape: {text_emb.shape}")

        # Compute CKA
        print("\n[3/3] Computing CKA...")
        metrics = compute_cka_metrics(vision_emb, text_emb)
        metrics["model_name"] = config["display_name"]
        metrics["vision_encoder"] = config["vision_encoder"]
        metrics["vision_model_id"] = config["vision_model_id"]
        metrics["llm_model_id"] = config["llm_model_id"]

        results[model_key] = metrics

        print(f"\n  Results for {config['display_name']}:")
        print(f"    CKA (linear): {metrics['cka_linear']:.4f}")
        print(f"    CKA (RBF):    {metrics['cka_rbf']:.4f}")
        print(f"    CKA (raw):    {metrics['cka_linear_raw']:.4f}")
        print(f"    Cosine sim:   {metrics['cosine_similarity']:.4f}")

        # Save intermediate results
        output_file = os.path.join(args.output_dir, "pretrained_cka.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  (Intermediate save: {output_file})")

        # Free vision encoder memory (LLM stays cached)
        gc.collect()
        torch.cuda.empty_cache()

    # Save final results
    output_file = os.path.join(args.output_dir, "pretrained_cka.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print("PRETRAINED CKA SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'CKA(lin)':>10} {'CKA(rbf)':>10} {'Cos':>8}")
    print("-" * 60)
    for key, m in results.items():
        print(f"{m['model_name']:<30} {m['cka_linear']:>10.4f} {m['cka_rbf']:>10.4f} {m['cosine_similarity']:>8.4f}")


if __name__ == "__main__":
    main()
