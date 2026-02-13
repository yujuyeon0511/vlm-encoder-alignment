#!/usr/bin/env python3
"""Gradio web UI for VLM Encoder Alignment Toolkit.

Launch: python app.py
Opens a browser with 5 tabs: Speed, Attention, Alignment, Embedding, E2E.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import numpy as np
from PIL import Image


# ============================================================
# Tab 1: Inference Speed Benchmark
# ============================================================

def run_speed_tab(encoders, batch_sizes_str, device):
    """Run speed benchmark and return dashboard image."""
    from vlm_alignment.analysis.speed_benchmark import InferenceSpeedBenchmark
    from vlm_alignment.visualization.speed_plots import plot_full_benchmark_dashboard

    encoder_list = [e.strip().lower() for e in encoders if e]
    if not encoder_list:
        encoder_list = ["clip", "siglip", "dinov2"]

    try:
        batch_sizes = [int(b.strip()) for b in batch_sizes_str.split(",")]
    except (ValueError, AttributeError):
        batch_sizes = [1, 4, 8, 16]

    bench = InferenceSpeedBenchmark(device=device, warmup_runs=2, benchmark_runs=5)
    result = bench.run_full_benchmark(encoder_names=encoder_list, batch_sizes=batch_sizes)

    fig = plot_full_benchmark_dashboard(result)
    path = os.path.join(tempfile.mkdtemp(), "speed_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)

    return path, result.summary_table()


# ============================================================
# Tab 2: Attention Maps
# ============================================================

def run_attention_tab(image, text, encoders, device):
    """Generate attention comparison for uploaded image."""
    from vlm_alignment.visualization.attention_maps import plot_attention_comparison

    if image is None:
        from vlm_alignment.data.synthetic import DataGenerator
        gen = DataGenerator()
        image = gen.generate_bar_chart()
        text = text or "What is the highest value?"

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    text = text or "Describe this image."
    encoder_list = [e.strip().lower() for e in encoders if e in ("clip", "siglip")]
    if not encoder_list:
        encoder_list = ["clip", "siglip"]

    fig = plot_attention_comparison(image, text, encoder_names=encoder_list, device=device)
    path = os.path.join(tempfile.mkdtemp(), "attention.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)

    return path


# ============================================================
# Tab 3: Alignment Analysis
# ============================================================

def run_alignment_tab(encoders, llm_name, n_samples, data_root, device):
    """Run alignment analysis and return results."""
    from vlm_alignment.models.vision_encoders import VisionEncoderManager
    from vlm_alignment.models.llm_loaders import LLMManager
    from vlm_alignment.analysis.cka import CKA
    from vlm_alignment.analysis.alignment import AlignmentAnalyzer
    from vlm_alignment.data.dataset import VLMDataset
    from vlm_alignment.visualization.alignment_plots import plot_alignment_summary

    encoder_list = [e.strip().lower() for e in encoders if e]
    if not encoder_list:
        encoder_list = ["clip", "siglip", "dinov2"]

    dataset = VLMDataset(data_root=data_root or None)
    samples, labels = dataset.load_mixed(n_per_type=int(n_samples))
    images, texts = dataset.get_images_and_texts(samples)

    if not images:
        return None, "No data available."

    ve_mgr = VisionEncoderManager(device=device)
    v_embs = ve_mgr.extract_multi_encoder(encoder_list, images)
    ve_mgr.unload()

    llm_mgr = LLMManager(device=device)
    t_embs = llm_mgr.extract_text_embeddings(llm_name.strip().lower(), texts)
    llm_mgr.unload()

    analyzer = AlignmentAnalyzer(device=device)
    metrics = analyzer.compute_alignment_metrics(v_embs, t_embs)

    fig = plot_alignment_summary(metrics)
    path = os.path.join(tempfile.mkdtemp(), "alignment.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)

    summary = "Alignment Metrics:\n"
    for name, m in metrics.items():
        summary += (
            f"\n{name.upper()}:\n"
            f"  CKA (Linear): {m.cka_linear:.4f}\n"
            f"  CKA (RBF):    {m.cka_rbf:.4f}\n"
            f"  Proj MSE:     {m.projection_mse:.4f}\n"
            f"  Cosine:       {m.cosine_after_proj:.4f}\n"
        )

    return path, summary


# ============================================================
# Tab 4: Embedding Space
# ============================================================

def run_embedding_tab(encoders, method, n_samples, data_root, device):
    """Generate embedding space visualization."""
    from vlm_alignment.models.vision_encoders import VisionEncoderManager
    from vlm_alignment.data.dataset import VLMDataset
    from vlm_alignment.visualization.embedding_space import plot_multi_encoder_tsne

    encoder_list = [e.strip().lower() for e in encoders if e]
    if not encoder_list:
        encoder_list = ["clip", "siglip", "dinov2"]

    dataset = VLMDataset(data_root=data_root or None)
    samples, labels = dataset.load_mixed(n_per_type=int(n_samples))
    images, _ = dataset.get_images_and_texts(samples)

    if not images:
        return None

    mgr = VisionEncoderManager(device=device)
    embs = mgr.extract_multi_encoder(encoder_list, images)
    mgr.unload()

    fig = plot_multi_encoder_tsne(embs, labels=labels, method=method.lower())
    path = os.path.join(tempfile.mkdtemp(), f"embedding_{method}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)

    return path


# ============================================================
# Tab 5: CORAL Analysis
# ============================================================

def run_coral_tab(encoders, llm_name, n_samples, data_root, device):
    """Run CORAL alignment analysis and return dashboard + summary."""
    import gc
    import torch
    from vlm_alignment.models.vision_encoders import VisionEncoderManager
    from vlm_alignment.models.llm_loaders import LLMManager
    from vlm_alignment.data.dataset import VLMDataset
    from vlm_alignment.analysis.coral import CORALAnalyzer
    from vlm_alignment.visualization.coral_plots import plot_eas_dashboard, plot_cka_vs_coral

    encoder_list = [e.strip().lower() for e in encoders if e]
    if not encoder_list:
        encoder_list = ["clip", "siglip", "dinov2"]

    dataset = VLMDataset(data_root=data_root or None)
    samples, labels = dataset.load_mixed(n_per_type=int(n_samples))
    images, texts = dataset.get_images_and_texts(samples)

    if not images:
        return None, None, "No data available."

    ve_mgr = VisionEncoderManager(device=device)
    v_embs = ve_mgr.extract_multi_encoder(encoder_list, images)
    ve_mgr.unload()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    llm_mgr = LLMManager(device=device)
    t_embs = llm_mgr.extract_text_embeddings(llm_name.strip().lower(), texts)
    llm_mgr.unload()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    analyzer = CORALAnalyzer()
    results = analyzer.full_analysis(v_embs, t_embs, llm_name, labels)

    # EAS dashboard
    eas_fig = plot_eas_dashboard(results["eas"])
    eas_path = os.path.join(tempfile.mkdtemp(), "eas_dashboard.png")
    eas_fig.savefig(eas_path, dpi=150, bbox_inches="tight", facecolor="white")

    # CKA vs CORAL scatter
    scatter_fig = plot_cka_vs_coral(results["cross_modal"])
    scatter_path = os.path.join(tempfile.mkdtemp(), "cka_vs_coral.png")
    scatter_fig.savefig(scatter_path, dpi=150, bbox_inches="tight", facecolor="white")

    import matplotlib.pyplot as plt
    plt.close(eas_fig)
    plt.close(scatter_fig)

    # Summary text
    summary = "CORAL Alignment Analysis:\n"
    for enc_name, r in results["cross_modal"].items():
        summary += (
            f"\n{enc_name.upper()}:\n"
            f"  CKA (Linear):     {r.cka_linear:.4f}\n"
            f"  CORAL Distance:   {r.coral.coral_distance:.4f}\n"
            f"  CORAL Similarity: {r.coral.coral_similarity:.4f}\n"
            f"  Spectral Div:     {r.coral.spectral_divergence:.4f}\n"
        )
    summary += "\nEnhanced Alignment Score (EAS):\n"
    for enc_name, r in results["eas"].items():
        summary += f"  {enc_name.upper()}: {r.eas:.4f} (CKA={r.cka:.3f}, CORAL={r.coral_score:.3f}, Disc={r.discriminability:.3f})\n"

    return eas_path, scatter_path, summary


# ============================================================
# Tab 6: E2E Validation
# ============================================================

def run_e2e_tab(encoders, llm_name, n_samples, data_root, device):
    """Run E2E validation experiment."""
    from vlm_alignment.experiments.e2e_validation import run_e2e_validation

    encoder_list = [e.strip().lower() for e in encoders if e]
    if not encoder_list:
        encoder_list = ["clip", "siglip", "dinov2"]

    tmpdir = tempfile.mkdtemp()
    results = run_e2e_validation(
        encoder_names=encoder_list,
        llm_name=llm_name.strip().lower(),
        n_samples=int(n_samples),
        data_root=data_root or None,
        output_dir=tmpdir,
        device=device,
    )

    summary = "E2E Validation Results:\n"
    for enc, metrics in results.items():
        if isinstance(metrics, dict):
            summary += (
                f"\n{enc.upper()}:\n"
                f"  CKA:      {metrics.get('cka', 'N/A'):.4f}\n"
                f"  Recall@1: {metrics.get('recall_1', 'N/A'):.3f}\n"
                f"  Recall@5: {metrics.get('recall_5', 'N/A'):.3f}\n"
                f"  MRR:      {metrics.get('mrr', 'N/A'):.3f}\n"
            )

    cka_path = os.path.join(tmpdir, "e2e_cka.png")
    return cka_path if os.path.exists(cka_path) else None, summary


# ============================================================
# App Assembly
# ============================================================

def build_app():
    import torch
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(title="VLM Encoder Alignment Toolkit", theme=gr.themes.Soft()) as app:
        gr.Markdown("# VLM Encoder Alignment Toolkit")
        gr.Markdown("Analyze how vision encoders align with LLMs through visual tools.")

        with gr.Tab("Inference Speed"):
            gr.Markdown("### Benchmark vision encoder inference speed")
            with gr.Row():
                speed_enc = gr.CheckboxGroup(
                    ["clip", "siglip", "dinov2", "internvit", "paligemma"],
                    value=["clip", "siglip", "dinov2"], label="Encoders"
                )
                speed_bs = gr.Textbox(value="1,4,8,16", label="Batch sizes (comma-separated)")
                speed_device = gr.Radio(["cuda", "cpu"], value=default_device, label="Device")
            speed_btn = gr.Button("Run Benchmark", variant="primary")
            speed_img = gr.Image(label="Dashboard")
            speed_txt = gr.Textbox(label="Summary", lines=15)
            speed_btn.click(run_speed_tab, [speed_enc, speed_bs, speed_device], [speed_img, speed_txt])

        with gr.Tab("Attention Maps"):
            gr.Markdown("### Visualize where encoders focus on images")
            with gr.Row():
                att_img = gr.Image(type="pil", label="Upload Image (optional)")
                att_text = gr.Textbox(value="Describe this image.", label="Query text")
            att_enc = gr.CheckboxGroup(["clip", "siglip"], value=["clip", "siglip"], label="Encoders")
            att_device = gr.Radio(["cuda", "cpu"], value=default_device, label="Device")
            att_btn = gr.Button("Generate Attention Maps", variant="primary")
            att_out = gr.Image(label="Attention Comparison")
            att_btn.click(run_attention_tab, [att_img, att_text, att_enc, att_device], att_out)

        with gr.Tab("Alignment Analysis"):
            gr.Markdown("### CKA alignment between encoders and LLM")
            with gr.Row():
                align_enc = gr.CheckboxGroup(
                    ["clip", "siglip", "dinov2"], value=["clip", "siglip", "dinov2"], label="Encoders"
                )
                align_llm = gr.Dropdown(["llama", "llama3", "qwen", "gemma3", "internlm"],
                                        value="llama", label="Target LLM")
            with gr.Row():
                align_n = gr.Slider(5, 100, value=10, step=5, label="Samples per type")
                align_data = gr.Textbox(value="", label="Data root (empty=sample data)")
                align_device = gr.Radio(["cuda", "cpu"], value=default_device, label="Device")
            align_btn = gr.Button("Analyze Alignment", variant="primary")
            align_img = gr.Image(label="Alignment Metrics")
            align_txt = gr.Textbox(label="Details", lines=15)
            align_btn.click(run_alignment_tab,
                            [align_enc, align_llm, align_n, align_data, align_device],
                            [align_img, align_txt])

        with gr.Tab("Embedding Space"):
            gr.Markdown("### t-SNE / UMAP visualization of encoder embedding spaces")
            with gr.Row():
                emb_enc = gr.CheckboxGroup(
                    ["clip", "siglip", "dinov2"], value=["clip", "siglip", "dinov2"], label="Encoders"
                )
                emb_method = gr.Radio(["tsne", "umap"], value="tsne", label="Method")
            with gr.Row():
                emb_n = gr.Slider(5, 100, value=10, step=5, label="Samples per type")
                emb_data = gr.Textbox(value="", label="Data root (empty=sample data)")
                emb_device = gr.Radio(["cuda", "cpu"], value=default_device, label="Device")
            emb_btn = gr.Button("Visualize Embeddings", variant="primary")
            emb_img = gr.Image(label="Embedding Space")
            emb_btn.click(run_embedding_tab, [emb_enc, emb_method, emb_n, emb_data, emb_device], emb_img)

        with gr.Tab("CORAL Analysis"):
            gr.Markdown("### Deep CORAL Alignment: CKA + CORAL + Discriminability")
            gr.Markdown("Compares covariance structures between vision and text embeddings. "
                        "Unlike CKA alone, CORAL captures distributional alignment.")
            with gr.Row():
                coral_enc = gr.CheckboxGroup(
                    ["clip", "siglip", "dinov2"], value=["clip", "siglip", "dinov2"], label="Encoders"
                )
                coral_llm = gr.Dropdown(["llama", "llama3", "qwen", "gemma3", "internlm"],
                                        value="llama", label="Target LLM")
            with gr.Row():
                coral_n = gr.Slider(5, 100, value=10, step=5, label="Samples per type")
                coral_data = gr.Textbox(value="", label="Data root (empty=sample data)")
                coral_device = gr.Radio(["cuda", "cpu"], value=default_device, label="Device")
            coral_btn = gr.Button("Run CORAL Analysis", variant="primary")
            with gr.Row():
                coral_eas_img = gr.Image(label="EAS Dashboard")
                coral_scatter_img = gr.Image(label="CKA vs CORAL")
            coral_txt = gr.Textbox(label="Details", lines=15)
            coral_btn.click(run_coral_tab,
                            [coral_enc, coral_llm, coral_n, coral_data, coral_device],
                            [coral_eas_img, coral_scatter_img, coral_txt])

        with gr.Tab("E2E Validation"):
            gr.Markdown("### CKA vs Task Performance (Paradox Detection)")
            with gr.Row():
                e2e_enc = gr.CheckboxGroup(
                    ["clip", "siglip", "dinov2"], value=["clip", "siglip", "dinov2"], label="Encoders"
                )
                e2e_llm = gr.Dropdown(["llama", "llama3", "qwen", "gemma3", "internlm"],
                                      value="llama", label="Target LLM")
            with gr.Row():
                e2e_n = gr.Slider(5, 100, value=10, step=5, label="Samples per type")
                e2e_data = gr.Textbox(value="", label="Data root (empty=sample data)")
                e2e_device = gr.Radio(["cuda", "cpu"], value=default_device, label="Device")
            e2e_btn = gr.Button("Run E2E Validation", variant="primary")
            e2e_img = gr.Image(label="Results")
            e2e_txt = gr.Textbox(label="Details", lines=15)
            e2e_btn.click(run_e2e_tab,
                          [e2e_enc, e2e_llm, e2e_n, e2e_data, e2e_device],
                          [e2e_img, e2e_txt])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
