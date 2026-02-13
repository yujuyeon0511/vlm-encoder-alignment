"""CORAL-Enhanced Alignment Experiment.

Runs Deep CORAL analysis comparing vision encoder - LLM alignment,
including cross-modal, intra-modal, and EAS scoring.

Usage:
    python cli.py coral --encoders clip siglip dinov2 --llms llama
"""

import os
import gc
import numpy as np
from typing import List, Optional


def run_coral_analysis(
    encoder_names: Optional[List[str]] = None,
    llm_name: str = "llama",
    n_samples: int = 30,
    data_root: Optional[str] = None,
    output_dir: str = "outputs",
    device: Optional[str] = None,
) -> dict:
    """Run full CORAL alignment analysis.

    Args:
        encoder_names: Vision encoders to compare
        llm_name: Target LLM
        n_samples: Samples per data type
        data_root: Path to real dataset (None = sample data)
        output_dir: Output directory for plots
        device: cuda or cpu

    Returns:
        Analysis results dict with cross_modal, intra_modal, eas
    """
    import torch
    from vlm_alignment.models.vision_encoders import VisionEncoderManager
    from vlm_alignment.models.llm_loaders import LLMManager
    from vlm_alignment.data.dataset import VLMDataset
    from vlm_alignment.analysis.coral import CORALAnalyzer
    from vlm_alignment.visualization.coral_plots import plot_coral_full_dashboard

    if encoder_names is None:
        encoder_names = ["clip", "siglip", "dinov2"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading dataset...")
    dataset = VLMDataset(data_root=data_root)
    samples, labels = dataset.load_mixed(n_per_type=n_samples)
    images, texts = dataset.get_images_and_texts(samples)

    if not images:
        print("No data available. Using synthetic data.")
        from vlm_alignment.data.synthetic import DataGenerator
        gen = DataGenerator()
        images = [gen.generate_bar_chart() for _ in range(3)]
        texts = ["What is the highest value?"] * 3
        labels = np.array([0, 0, 0])

    print(f"Loaded {len(images)} samples")

    # Extract vision embeddings
    print(f"\nExtracting vision embeddings...")
    ve_mgr = VisionEncoderManager(device=device)
    vision_embeddings = ve_mgr.extract_multi_encoder(encoder_names, images)
    ve_mgr.unload()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract text embeddings
    print(f"\nExtracting text embeddings ({llm_name})...")
    llm_mgr = LLMManager(device=device)
    text_embeddings = llm_mgr.extract_text_embeddings(llm_name, texts)
    llm_mgr.unload()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run CORAL analysis
    print(f"\nRunning CORAL analysis...")
    analyzer = CORALAnalyzer()
    results = analyzer.full_analysis(
        vision_embeddings=vision_embeddings,
        text_embeddings=text_embeddings,
        llm_name=llm_name,
        labels=labels,
    )

    # Print results
    print("\n" + "=" * 60)
    print("CORAL Analysis Results")
    print("=" * 60)

    print("\n--- Cross-Modal: Vision vs Text ---")
    for enc_name, r in results["cross_modal"].items():
        print(f"\n{enc_name.upper()}:")
        print(f"  CKA (Linear):       {r.cka_linear:.4f}")
        print(f"  CORAL Distance:     {r.coral.coral_distance:.4f}")
        print(f"  CORAL Similarity:   {r.coral.coral_similarity:.4f}")
        print(f"  Spectral Div:       {r.coral.spectral_divergence:.4f}")
        print(f"  Mean Distance:      {r.coral.mean_distance:.4f}")

    if results["intra_modal"]:
        print("\n--- Intra-Modal: Encoder vs Encoder ---")
        for r in results["intra_modal"]:
            print(f"\n{r.name_a.upper()} vs {r.name_b.upper()}:")
            print(f"  CKA:              {r.cka_linear:.4f}")
            print(f"  CORAL Similarity: {r.coral.coral_similarity:.4f}")

    print("\n--- Enhanced Alignment Score (EAS) ---")
    for enc_name, r in results["eas"].items():
        print(f"\n{enc_name.upper()}:")
        print(f"  CKA:              {r.cka:.4f}")
        print(f"  CORAL Score:      {r.coral_score:.4f}")
        print(f"  Discriminability: {r.discriminability:.4f}")
        print(f"  EAS:              {r.eas:.4f}")

    # Ranking comparison
    eas_ranking = sorted(results["eas"].keys(),
                         key=lambda n: results["eas"][n].eas, reverse=True)
    cka_ranking = sorted(results["eas"].keys(),
                         key=lambda n: results["eas"][n].cka, reverse=True)

    print(f"\nRanking by CKA:  {' > '.join(e.upper() for e in cka_ranking)}")
    print(f"Ranking by EAS:  {' > '.join(e.upper() for e in eas_ranking)}")

    if cka_ranking[0] != eas_ranking[0]:
        print(f"\n** CKA and EAS disagree on best encoder! **")
        print(f"   CKA best: {cka_ranking[0].upper()}, EAS best: {eas_ranking[0].upper()}")
        print(f"   This demonstrates the CKA-Performance Paradox.")

    # Generate plots
    print(f"\nGenerating plots...")
    figures = plot_coral_full_dashboard(results, output_dir=output_dir)
    print(f"\nSaved {len(figures)} plots to {output_dir}/")

    return results
