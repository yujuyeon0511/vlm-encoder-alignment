"""ELAS (Encoder-LLM Alignment Score) experiment.

Computes the composite ELAS metric for all encoder x LLM combinations
to find the optimal pairing.
"""

from typing import List, Optional, Dict

from vlm_alignment.config import get_output_dir, get_device
from vlm_alignment.models.vision_encoders import VisionEncoderManager
from vlm_alignment.models.llm_loaders import LLMManager
from vlm_alignment.analysis.alignment import AlignmentAnalyzer
from vlm_alignment.data.dataset import VLMDataset
from vlm_alignment.visualization.alignment_plots import plot_elas_results


def run_elas_experiment(
    encoder_names: List[str] = None,
    llm_names: List[str] = None,
    n_samples: int = 30,
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Run ELAS scoring across encoder x LLM matrix.

    ELAS = alpha*CKA + beta*(1-MSE) + gamma*(1-GenGap) + delta*CosSim

    Args:
        encoder_names: Vision encoders to evaluate
        llm_names: Target LLMs
        n_samples: Samples per data type
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
    print("ELAS (Encoder-LLM Alignment Score) Experiment")
    print(f"Encoders: {encoder_names}")
    print(f"LLMs: {llm_names}")
    print("=" * 60)

    # Load data
    dataset = VLMDataset(data_root=data_root)
    samples, labels = dataset.load_mixed(n_per_type=n_samples)
    images, texts = dataset.get_images_and_texts(samples)

    if not images:
        print("No data available.")
        return {}

    # Extract vision embeddings
    ve_mgr = VisionEncoderManager(device=device)
    vision_embs = ve_mgr.extract_multi_encoder(encoder_names, images)
    ve_mgr.unload()

    # Extract text embeddings for each LLM
    llm_mgr = LLMManager(device=device)
    text_embs = {}
    for llm in llm_names:
        text_embs[llm] = llm_mgr.extract_text_embeddings(llm, texts)
        llm_mgr.unload(llm)

    # Compute ELAS matrix
    analyzer = AlignmentAnalyzer(device=device)
    elas_scores = analyzer.compute_elas_matrix(vision_embs, text_embs)

    # Visualize
    plot_elas_results(elas_scores, output_path=os.path.join(output_dir, "elas_matrix.png"))

    # Find optimal combinations
    print("\n" + "=" * 60)
    print("Optimal Combinations:")
    best_by_llm = {}
    for llm in llm_names:
        llm_scores = [s for s in elas_scores if s.llm == llm]
        if llm_scores:
            best = max(llm_scores, key=lambda s: s.score)
            best_by_llm[llm] = best
            print(f"  {llm}: {best.encoder} (ELAS={best.score:.4f})")

    return {
        "elas_scores": elas_scores,
        "best_by_llm": best_by_llm,
    }
