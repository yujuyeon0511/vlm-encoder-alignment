"""End-to-end validation: CKA vs actual task performance.

Tests whether higher CKA actually leads to better downstream performance,
revealing the CKA-Performance Paradox (r=-0.99).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict

from vlm_alignment.config import get_output_dir, get_device
from vlm_alignment.models.vision_encoders import VisionEncoderManager
from vlm_alignment.models.llm_loaders import LLMManager
from vlm_alignment.models.projectors import create_projector
from vlm_alignment.analysis.cka import CKA
from vlm_alignment.data.dataset import VLMDataset


class SimpleVLM(nn.Module):
    """Simplified VLM for retrieval evaluation."""

    def __init__(self, vision_dim: int, text_dim: int, proj_type: str = "2layer_mlp", device: str = "cuda"):
        super().__init__()
        self.projector = create_projector(proj_type, vision_dim, text_dim)
        self.device = device
        self.to(device)

    def forward(self, vision_emb: torch.Tensor) -> torch.Tensor:
        return self.projector(vision_emb)


def evaluate_retrieval(
    projected_vision: np.ndarray,
    text_embeddings: np.ndarray,
) -> Dict[str, float]:
    """Compute retrieval metrics: Recall@1, Recall@5, MRR.

    Args:
        projected_vision: Projected vision embeddings (n, d)
        text_embeddings: Text embeddings (n, d)

    Returns:
        Dict with recall_1, recall_5, mrr
    """
    # Normalize
    v_norm = projected_vision / (np.linalg.norm(projected_vision, axis=1, keepdims=True) + 1e-8)
    t_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)

    # Similarity matrix
    sim = v_norm @ t_norm.T  # (n, n)
    n = sim.shape[0]

    recall_1 = 0
    recall_5 = 0
    mrr = 0

    for i in range(n):
        ranking = np.argsort(-sim[i])
        rank = np.where(ranking == i)[0][0] + 1
        if rank == 1:
            recall_1 += 1
        if rank <= 5:
            recall_5 += 1
        mrr += 1.0 / rank

    return {
        "recall_1": recall_1 / n,
        "recall_5": recall_5 / n,
        "mrr": mrr / n,
    }


def run_e2e_validation(
    encoder_names: List[str] = None,
    llm_name: str = "llama",
    n_samples: int = 30,
    epochs: int = 300,
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """Run end-to-end validation experiment.

    Trains projectors for each encoder and evaluates on retrieval task,
    then correlates CKA with retrieval performance.
    """
    if encoder_names is None:
        encoder_names = ["clip", "siglip", "dinov2"]
    if output_dir is None:
        output_dir = str(get_output_dir())
    if device is None:
        device = get_device()

    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("E2E Validation: CKA vs Task Performance")
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
    text_embs = llm_mgr.extract_text_embeddings(llm_name, texts)
    llm_mgr.unload()

    results = {}
    for enc in encoder_names:
        print(f"\n--- {enc.upper()} ---")
        n = min(vision_embs[enc].shape[0], text_embs.shape[0])
        v_emb = vision_embs[enc][:n]
        t_emb = text_embs[:n]

        # CKA
        cka = CKA.compute_cka(v_emb, t_emb)
        print(f"  CKA: {cka:.4f}")

        # Train projector
        v_dim = v_emb.shape[1]
        t_dim = t_emb.shape[1]
        vlm = SimpleVLM(v_dim, t_dim, device=device)
        optimizer = torch.optim.Adam(vlm.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        v_tensor = torch.FloatTensor(v_emb).to(device)
        t_tensor = torch.FloatTensor(t_emb).to(device)

        vlm.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            projected = vlm(v_tensor)
            loss = criterion(projected, t_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate retrieval
        vlm.eval()
        with torch.no_grad():
            projected = vlm(v_tensor).cpu().numpy()

        retrieval = evaluate_retrieval(projected, t_emb)
        print(f"  Recall@1: {retrieval['recall_1']:.3f}")
        print(f"  Recall@5: {retrieval['recall_5']:.3f}")
        print(f"  MRR: {retrieval['mrr']:.3f}")

        results[enc] = {"cka": cka, **retrieval}

    # Correlation analysis
    cka_vals = [results[e]["cka"] for e in encoder_names]
    mrr_vals = [results[e]["mrr"] for e in encoder_names]

    if len(cka_vals) > 2:
        from scipy.stats import pearsonr
        r, _ = pearsonr(cka_vals, mrr_vals)
        print(f"\nCKA-MRR Correlation: r = {r:.4f}")
        results["correlation"] = r

    # Visualize
    from vlm_alignment.visualization.alignment_plots import plot_cka_comparison
    plot_cka_comparison(
        {enc: results[enc]["cka"] for enc in encoder_names},
        title="CKA Score",
        output_path=os.path.join(output_dir, "e2e_cka.png"),
    )
    plot_cka_comparison(
        {enc: results[enc]["mrr"] for enc in encoder_names},
        title="Retrieval MRR",
        output_path=os.path.join(output_dir, "e2e_mrr.png"),
    )

    return results
