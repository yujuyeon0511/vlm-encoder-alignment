"""Encoder comparison experiment.

Consolidates real_data_experiment.py and multi_llm_experiment.py:
compares vision encoders across data types and LLMs using the
unified models/ package (no more duplicated loading code).
"""

import numpy as np
from typing import List, Optional, Dict

from vlm_alignment.config import get_output_dir, get_device
from vlm_alignment.models.vision_encoders import VisionEncoderManager
from vlm_alignment.models.llm_loaders import LLMManager
from vlm_alignment.models.projectors import train_projector
from vlm_alignment.analysis.cka import CKA
from vlm_alignment.data.dataset import VLMDataset
from vlm_alignment.visualization.alignment_plots import (
    plot_cka_comparison, plot_cka_by_data_type, plot_projector_comparison,
    plot_alignment_summary,
)


def run_encoder_comparison(
    encoder_names: List[str] = None,
    llm_name: str = "llama",
    data_types: List[str] = None,
    n_samples: int = 30,
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Run encoder comparison experiment.

    Compares vision encoders on CKA alignment with a target LLM,
    broken down by data type (chart, table, text).

    Args:
        encoder_names: Encoders to compare (default: clip, siglip, dinov2)
        llm_name: Target LLM for text embeddings
        data_types: Data types to evaluate
        n_samples: Samples per data type
        data_root: Override data path
        output_dir: Output directory for plots
        device: torch device
    """
    if encoder_names is None:
        encoder_names = ["clip", "siglip", "dinov2"]
    if data_types is None:
        data_types = ["chart", "table", "text"]
    if output_dir is None:
        output_dir = str(get_output_dir())
    if device is None:
        device = get_device()

    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Encoder Comparison Experiment")
    print(f"Encoders: {encoder_names}")
    print(f"Target LLM: {llm_name}")
    print(f"Data types: {data_types}")
    print("=" * 60)

    # Load data
    dataset = VLMDataset(data_root=data_root)
    ve_mgr = VisionEncoderManager(device=device)
    llm_mgr = LLMManager(device=device)

    overall_cka = {enc: [] for enc in encoder_names}
    per_type_cka = {dt: {} for dt in data_types}

    for dt in data_types:
        print(f"\n--- Data type: {dt} ---")
        samples = dataset.load_samples(dt, n_samples)
        images, texts = dataset.get_images_and_texts(samples)

        if not images:
            print(f"  No samples available for {dt}, skipping.")
            continue

        # Extract embeddings
        vision_embs = ve_mgr.extract_multi_encoder(encoder_names, images)
        text_embs = llm_mgr.extract_text_embeddings(llm_name, texts)

        # Compute CKA per encoder
        for enc in encoder_names:
            n = min(vision_embs[enc].shape[0], text_embs.shape[0])
            cka = CKA.compute_cka(vision_embs[enc][:n], text_embs[:n])
            overall_cka[enc].append(cka)
            per_type_cka[dt][enc] = cka
            print(f"  {enc}: CKA = {cka:.4f}")

        # Free memory between data types
        ve_mgr.unload()

    # Compute overall averages
    avg_cka = {enc: np.mean(scores) if scores else 0 for enc, scores in overall_cka.items()}

    print("\n" + "=" * 60)
    print("RESULTS: Overall CKA")
    for enc, score in sorted(avg_cka.items(), key=lambda x: -x[1]):
        print(f"  {enc}: {score:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_cka_comparison(avg_cka, output_path=os.path.join(output_dir, "cka_comparison.png"))
    plot_cka_by_data_type(per_type_cka, output_path=os.path.join(output_dir, "cka_by_data_type.png"))

    # Projector comparison
    print("\nRunning projector comparison...")
    dataset_all = dataset.load_mixed(n_per_type=n_samples, data_types=data_types)
    if dataset_all[0]:
        images_all, _ = dataset.get_images_and_texts(dataset_all[0])
        texts_all = [s.question for s in dataset_all[0]]

        ve_mgr_fresh = VisionEncoderManager(device=device)
        v_embs = ve_mgr_fresh.extract_multi_encoder(encoder_names, images_all)
        t_embs = llm_mgr.extract_text_embeddings(llm_name, texts_all)

        proj_results = {}
        for enc in encoder_names:
            proj_results[enc] = {}
            n = min(v_embs[enc].shape[0], t_embs.shape[0])
            for pt in ["linear", "mlp", "2layer_mlp"]:
                result = train_projector(
                    v_embs[enc][:n], t_embs[:n],
                    projection_type=pt, device=device, test_ratio=0.2,
                )
                proj_results[enc][pt] = result.train_loss
                print(f"  {enc}/{pt}: train_loss={result.train_loss:.4f}, "
                      f"test_loss={result.test_loss:.4f if result.test_loss else 'N/A'}")

        plot_projector_comparison(proj_results, output_path=os.path.join(output_dir, "projector_comparison.png"))

    llm_mgr.unload()

    return {
        "overall_cka": avg_cka,
        "per_type_cka": per_type_cka,
        "projector_results": proj_results if dataset_all[0] else {},
    }


def run_multi_llm_comparison(
    encoder_names: List[str] = None,
    llm_names: List[str] = None,
    n_samples: int = 30,
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Run multi-LLM comparison to find optimal encoder per LLM.

    Tests all encoder x LLM combinations and finds the best
    encoder for each target LLM.
    """
    if encoder_names is None:
        encoder_names = ["clip", "siglip", "dinov2"]
    if llm_names is None:
        llm_names = ["llama", "qwen"]
    if output_dir is None:
        output_dir = str(get_output_dir())
    if device is None:
        device = get_device()

    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-LLM Comparison")
    print(f"Encoders: {encoder_names}")
    print(f"LLMs: {llm_names}")
    print("=" * 60)

    dataset = VLMDataset(data_root=data_root)
    samples, labels = dataset.load_mixed(n_per_type=n_samples)
    images, texts = dataset.get_images_and_texts(samples)

    if not images:
        print("No data available.")
        return {}

    ve_mgr = VisionEncoderManager(device=device)
    vision_embs = ve_mgr.extract_multi_encoder(encoder_names, images)
    ve_mgr.unload()

    llm_mgr = LLMManager(device=device)
    cka_matrix = np.zeros((len(encoder_names), len(llm_names)))

    for j, llm in enumerate(llm_names):
        text_embs = llm_mgr.extract_text_embeddings(llm, texts)
        for i, enc in enumerate(encoder_names):
            n = min(vision_embs[enc].shape[0], text_embs.shape[0])
            cka = CKA.compute_cka(vision_embs[enc][:n], text_embs[:n])
            cka_matrix[i, j] = cka
            print(f"  {enc} x {llm}: CKA = {cka:.4f}")
        llm_mgr.unload(llm)

    from vlm_alignment.visualization.alignment_plots import plot_cka_heatmap
    plot_cka_heatmap(
        encoder_names, llm_names, cka_matrix,
        title="Encoder x LLM CKA Matrix",
        output_path=os.path.join(output_dir, "multi_llm_cka_matrix.png"),
    )

    # Find best encoder per LLM
    print("\nOptimal Encoder per LLM:")
    for j, llm in enumerate(llm_names):
        best_idx = np.argmax(cka_matrix[:, j])
        print(f"  {llm}: {encoder_names[best_idx]} (CKA={cka_matrix[best_idx, j]:.4f})")

    return {"cka_matrix": cka_matrix, "encoders": encoder_names, "llms": llm_names}
