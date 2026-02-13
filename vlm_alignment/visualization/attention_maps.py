"""Attention map visualization for vision encoders.

Based on attention_visualization_v3.py - shows patch-text similarity
heatmaps overlaid on input images, with entropy comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Optional

from vlm_alignment.models.vision_encoders import VisionEncoderManager
from vlm_alignment.visualization.plot_style import (
    apply_style, get_model_color, style_axis, create_figure, save_figure,
)


def compute_patch_text_similarity(
    encoder_name: str,
    image: Image.Image,
    text: str,
    device: str = "cuda",
) -> Dict:
    """Compute per-patch text similarity for a vision encoder.

    Args:
        encoder_name: 'clip' or 'siglip'
        image: Input image
        text: Query text
        device: torch device

    Returns:
        Dict with 'similarity_map' (2D array), 'grid_size', 'entropy'
    """
    mgr = VisionEncoderManager(device=device)
    model, processor = mgr.load(encoder_name, with_attention=True)

    with torch.no_grad():
        if encoder_name == "clip":
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            vision_out = model.vision_model(
                pixel_values=image_inputs["pixel_values"],
                output_hidden_states=True,
            )
            patch_emb = vision_out.last_hidden_state[0, 1:]
            patch_proj = model.visual_projection(patch_emb)
            patch_proj = F.normalize(patch_proj, dim=-1)

            text_inputs = processor(
                text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(device)
            text_out = model.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            text_emb = model.text_projection(text_out.pooler_output)
            text_emb = F.normalize(text_emb, dim=-1)

            sims = (patch_proj @ text_emb.T).squeeze().cpu().numpy()
            grid_size = int(np.sqrt(len(sims)))

        elif encoder_name == "siglip":
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
            pixel_values = image_inputs.get("pixel_values", image_inputs.get("image")).to(device)

            vision_out = model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            patch_emb = vision_out.last_hidden_state[0]
            patch_proj = F.normalize(patch_emb, dim=-1)

            text_out = model.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )
            text_emb = text_out.last_hidden_state[:, 0]
            text_emb = F.normalize(text_emb, dim=-1)

            sims = (patch_proj @ text_emb.T).squeeze().cpu().numpy()
            grid_size = int(np.sqrt(len(sims)))

        else:
            raise ValueError(f"Patch-text similarity not supported for {encoder_name}")

    sim_map = sims[: grid_size * grid_size].reshape(grid_size, grid_size)

    # Compute attention entropy
    probs = np.exp(sims - sims.max())
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return {"similarity_map": sim_map, "grid_size": grid_size, "entropy": entropy}


def plot_attention_heatmap(
    image: Image.Image,
    similarity_map: np.ndarray,
    encoder_name: str,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay attention heatmap on image.

    Args:
        image: Original PIL image
        similarity_map: 2D attention/similarity map
        encoder_name: For color scheme
        title: Optional title

    Returns:
        matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Resize similarity map to image size
    img_array = np.array(image.resize((224, 224)))
    sim_resized = np.array(
        Image.fromarray(
            ((similarity_map - similarity_map.min())
             / (similarity_map.max() - similarity_map.min() + 1e-8) * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)
    )

    ax.imshow(img_array)
    ax.imshow(sim_resized, alpha=0.5, cmap="jet")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    return ax


def plot_attention_comparison(
    image: Image.Image,
    text: str,
    encoder_names: List[str] = None,
    output_path: Optional[str] = None,
    device: str = "cuda",
) -> plt.Figure:
    """Compare attention maps across encoders for a single image.

    Args:
        image: Input image
        text: Query text
        encoder_names: Encoders to compare (default: clip, siglip)
        output_path: If set, save figure to this path

    Returns:
        matplotlib Figure
    """
    if encoder_names is None:
        encoder_names = ["clip", "siglip"]

    apply_style()
    fig, axes = plt.subplots(1, len(encoder_names) + 1, figsize=(5 * (len(encoder_names) + 1), 5))

    # Original image
    axes[0].imshow(np.array(image.resize((224, 224))))
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    entropies = {}
    for i, name in enumerate(encoder_names):
        result = compute_patch_text_similarity(name, image, text, device=device)
        plot_attention_heatmap(image, result["similarity_map"], name, title=name.upper(), ax=axes[i + 1])
        entropies[name] = result["entropy"]

    fig.suptitle(f'Query: "{text[:60]}..."' if len(text) > 60 else f'Query: "{text}"', fontsize=13)
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_entropy_comparison(
    images: List[Image.Image],
    texts: List[str],
    encoder_names: List[str] = None,
    output_path: Optional[str] = None,
    device: str = "cuda",
) -> plt.Figure:
    """Compare attention entropy distributions across encoders.

    Args:
        images: List of input images
        texts: Corresponding query texts
        encoder_names: Encoders to compare

    Returns:
        matplotlib Figure with entropy box plots
    """
    if encoder_names is None:
        encoder_names = ["clip", "siglip"]

    all_entropies = {name: [] for name in encoder_names}

    for img, txt in zip(images, texts):
        for name in encoder_names:
            result = compute_patch_text_similarity(name, img, txt, device=device)
            all_entropies[name].append(result["entropy"])

    fig, ax = create_figure()
    positions = range(len(encoder_names))
    colors = [get_model_color(n) for n in encoder_names]

    bp = ax.boxplot(
        [all_entropies[n] for n in encoder_names],
        positions=positions,
        patch_artist=True,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([n.upper() for n in encoder_names])
    style_axis(ax, title="Attention Entropy by Encoder", ylabel="Entropy")

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig
