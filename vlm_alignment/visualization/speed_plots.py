"""Speed benchmark visualization.

NEW module: Plots for latency, throughput, memory usage, and
speed vs accuracy tradeoffs.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from vlm_alignment.analysis.speed_benchmark import SpeedResult, FullBenchmarkResult
from vlm_alignment.visualization.plot_style import (
    apply_style, get_model_color, style_axis, create_figure, save_figure,
)


def plot_latency_comparison(
    results: List[SpeedResult],
    batch_size: int = 1,
    title: str = "Inference Latency Comparison",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing latency across models for a fixed batch size.

    Args:
        results: List of SpeedResult
        batch_size: Which batch size to plot
        title: Chart title
    """
    filtered = [r for r in results if r.batch_size == batch_size and r.component == "vision_encoder"]
    if not filtered:
        filtered = [r for r in results if r.batch_size == batch_size]

    fig, ax = create_figure()
    names = [r.model_name for r in filtered]
    latencies = [r.latency_ms for r in filtered]
    stds = [r.std_ms for r in filtered]
    colors = [get_model_color(n) for n in names]

    bars = ax.bar(names, latencies, yerr=stds, color=colors,
                  edgecolor="black", linewidth=1.5, capsize=5)

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")

    style_axis(ax, title=f"{title} (batch={batch_size})", ylabel="Latency (ms)")
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_throughput_by_batch(
    results: List[SpeedResult],
    component: str = "vision_encoder",
    title: str = "Throughput vs Batch Size",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Line chart of throughput across batch sizes.

    Args:
        results: List of SpeedResult
        component: Filter by component type
    """
    filtered = [r for r in results if r.component == component]
    models = sorted(set(r.model_name for r in filtered))

    fig, ax = create_figure()

    for model_name in models:
        model_results = sorted(
            [r for r in filtered if r.model_name == model_name],
            key=lambda x: x.batch_size,
        )
        bs = [r.batch_size for r in model_results]
        tp = [r.throughput for r in model_results]
        ax.plot(bs, tp, "o-", color=get_model_color(model_name),
                label=model_name.upper(), linewidth=2, markersize=8)

    ax.legend()
    style_axis(ax, title=title, xlabel="Batch Size", ylabel="Throughput (items/s)")
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_memory_comparison(
    results: List[SpeedResult],
    batch_size: int = 1,
    title: str = "GPU Memory Usage",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing GPU memory usage.

    Args:
        results: List of SpeedResult
        batch_size: Which batch size to show
    """
    filtered = [r for r in results if r.batch_size == batch_size and r.component == "vision_encoder"]
    if not filtered:
        filtered = [r for r in results if r.batch_size == batch_size]

    fig, ax = create_figure()
    names = [r.model_name for r in filtered]
    memory = [r.memory_mb for r in filtered]
    colors = [get_model_color(n) for n in names]

    bars = ax.bar(names, memory, color=colors, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, memory):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}MB", ha="center", va="bottom", fontsize=11, fontweight="bold")

    style_axis(ax, title=f"{title} (batch={batch_size})", ylabel="Memory (MB)")
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_speed_vs_accuracy(
    speed_results: List[SpeedResult],
    accuracy_scores: dict,
    batch_size: int = 1,
    title: str = "Speed vs Alignment Quality",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot: inference speed vs alignment quality.

    Args:
        speed_results: List of SpeedResult (vision_encoder component)
        accuracy_scores: {encoder_name: cka_score}
        batch_size: Which batch size
    """
    filtered = [r for r in speed_results
                if r.batch_size == batch_size and r.component == "vision_encoder"]

    fig, ax = create_figure()

    for r in filtered:
        if r.model_name in accuracy_scores:
            color = get_model_color(r.model_name)
            ax.scatter(r.latency_ms, accuracy_scores[r.model_name],
                       c=color, s=200, edgecolors="black", linewidth=1.5, zorder=5)
            ax.annotate(r.model_name.upper(),
                        (r.latency_ms, accuracy_scores[r.model_name]),
                        textcoords="offset points", xytext=(10, 10), fontsize=11)

    style_axis(ax, title=title, xlabel="Latency (ms)", ylabel="CKA Score")
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_full_benchmark_dashboard(
    benchmark: FullBenchmarkResult,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Full dashboard with latency, throughput, and memory plots.

    Args:
        benchmark: FullBenchmarkResult from speed benchmark
    """
    ve_results = [r for r in benchmark.results if r.component == "vision_encoder"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    apply_style()

    # Top-left: Latency at batch=1
    bs1 = [r for r in ve_results if r.batch_size == 1]
    if bs1:
        names = [r.model_name for r in bs1]
        vals = [r.latency_ms for r in bs1]
        colors = [get_model_color(n) for n in names]
        bars = axes[0, 0].bar(names, vals, color=colors, edgecolor="black")
        for b, v in zip(bars, vals):
            axes[0, 0].text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                            ha="center", fontsize=10, fontweight="bold")
        style_axis(axes[0, 0], title="Latency (batch=1)", ylabel="ms")

    # Top-right: Throughput curves
    models = sorted(set(r.model_name for r in ve_results))
    for m in models:
        mr = sorted([r for r in ve_results if r.model_name == m], key=lambda x: x.batch_size)
        axes[0, 1].plot([r.batch_size for r in mr], [r.throughput for r in mr],
                        "o-", color=get_model_color(m), label=m.upper(), linewidth=2)
    axes[0, 1].legend()
    style_axis(axes[0, 1], title="Throughput vs Batch Size", xlabel="Batch", ylabel="img/s")

    # Bottom-left: Memory
    if bs1:
        mem = [r.memory_mb for r in bs1]
        bars = axes[1, 0].bar(names, mem, color=colors, edgecolor="black")
        for b, v in zip(bars, mem):
            axes[1, 0].text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}",
                            ha="center", fontsize=10, fontweight="bold")
        style_axis(axes[1, 0], title="GPU Memory (batch=1)", ylabel="MB")

    # Bottom-right: Projector comparison
    proj_results = [r for r in benchmark.results if r.component == "projector" and r.batch_size == 8]
    if proj_results:
        pnames = [r.model_name for r in proj_results]
        plat = [r.latency_ms for r in proj_results]
        axes[1, 1].bar(pnames, plat, color="#9cb35e", edgecolor="black")
        for x_pos, val in enumerate(plat):
            axes[1, 1].text(x_pos, val + 0.01, f"{val:.2f}",
                            ha="center", fontsize=10, fontweight="bold")
        style_axis(axes[1, 1], title="Projector Latency (batch=8)", ylabel="ms")

    plt.suptitle("Inference Speed Benchmark Dashboard", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)
    return fig
