"""
Analyze correlation between CKA metrics and benchmark performance.

Combines:
  1. lmms-eval benchmark results (from outputs/benchmark/)
  2. Pretrained CKA scores (from outputs/cka/pretrained_cka.json)
  3. Internal CKA scores (from outputs/cka/internal_cka.json)

Produces:
  - Correlation analysis (Pearson/Spearman)
  - Scatter plots: CKA vs benchmark score
  - Heatmap: model x benchmark
  - Delta plot: pretrained vs post-projector CKA change

Usage:
    conda run -n docmllm python scripts/analyze_cka_vs_benchmark.py
    conda run -n docmllm python scripts/analyze_cka_vs_benchmark.py --benchmark_dir outputs/benchmark
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Constants
# ============================================================================
MODEL_ORDER = ["qwen25vl", "llava_ov", "llava_gemma_siglip2", "llava_gemma_clip"]
MODEL_DISPLAY = {
    "qwen25vl": "Qwen2.5-VL-7B",
    "llava_ov": "LLaVA-OV-Qwen2-7B",
    "llava_gemma_siglip2": "LLaVA-Gemma-SigLIP2",
    "llava_gemma_clip": "LLaVA-Gemma-CLIP",
}
MODEL_COLORS = {
    "qwen25vl": "#4C8BF5",
    "llava_ov": "#F5A623",
    "llava_gemma_siglip2": "#0ABAB5",
    "llava_gemma_clip": "#E94B3C",
}
MODEL_MARKERS = {
    "qwen25vl": "o",
    "llava_ov": "s",
    "llava_gemma_siglip2": "^",
    "llava_gemma_clip": "D",
}

BENCHMARK_DISPLAY = {
    "textvqa_val": "TextVQA",
    "chartqa": "ChartQA",
    "docvqa_val": "DocVQA",
    "mmbench_en_dev": "MMBench",
    "mmstar": "MMStar",
    "pope": "POPE",
}


# ============================================================================
# Data Loading
# ============================================================================
def load_benchmark_results(benchmark_dir: str) -> dict:
    """Load benchmark results from lmms-eval output directories."""
    results = {}

    for model_key in MODEL_ORDER:
        model_dir = os.path.join(benchmark_dir, model_key)
        if not os.path.isdir(model_dir):
            print(f"  Warning: {model_dir} not found, skipping {model_key}")
            continue

        # lmms-eval saves results in various formats, try to find them
        model_results = {}

        # Try: results.json in the model directory
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                if f == "results.json" or f.endswith("_results.json"):
                    fpath = os.path.join(root, f)
                    try:
                        with open(fpath) as fp:
                            data = json.load(fp)
                        # Extract scores from lmms-eval format
                        if "results" in data:
                            for task_key, task_data in data["results"].items():
                                # Normalize task name
                                task_name = task_key.split(",")[0].strip()
                                if isinstance(task_data, dict):
                                    # Get the main metric
                                    score = None
                                    for metric_key in ["acc", "exact_match", "accuracy",
                                                      "relaxed_accuracy", "score",
                                                      "anls", "f1"]:
                                        if metric_key in task_data:
                                            score = task_data[metric_key]
                                            break
                                    if score is None and task_data:
                                        # Take first numeric value
                                        for v in task_data.values():
                                            if isinstance(v, (int, float)):
                                                score = v
                                                break
                                    if score is not None:
                                        model_results[task_name] = float(score)
                        elif isinstance(data, dict):
                            # Simple key-value format
                            for k, v in data.items():
                                if isinstance(v, (int, float)):
                                    model_results[k] = float(v)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"  Warning: failed to parse {fpath}: {e}")

        if model_results:
            results[model_key] = model_results
            print(f"  Loaded {len(model_results)} benchmarks for {model_key}")

    return results


def load_cka_results(cka_dir: str) -> tuple:
    """Load pretrained and internal CKA results."""
    pretrained = {}
    internal = {}

    pretrained_path = os.path.join(cka_dir, "pretrained_cka.json")
    if os.path.exists(pretrained_path):
        with open(pretrained_path) as f:
            pretrained = json.load(f)
        print(f"  Loaded pretrained CKA for {len(pretrained)} models")

    internal_path = os.path.join(cka_dir, "internal_cka.json")
    if os.path.exists(internal_path):
        with open(internal_path) as f:
            internal = json.load(f)
        print(f"  Loaded internal CKA for {len(internal)} models")

    return pretrained, internal


# ============================================================================
# Analysis
# ============================================================================
def compute_correlations(benchmark_results: dict, cka_pretrained: dict,
                        cka_internal: dict) -> dict:
    """Compute correlations between CKA and benchmark scores."""
    correlations = {}

    # Get models that have both CKA and benchmark data
    common_models = sorted(
        set(benchmark_results.keys()) &
        set(cka_pretrained.keys())
    )

    if len(common_models) < 3:
        print(f"  Warning: Only {len(common_models)} models with complete data. "
              f"Need at least 3 for correlation.")

    if len(common_models) < 2:
        return correlations

    # Pretrained CKA vs each benchmark
    for metric_key in ["cka_linear", "cka_rbf"]:
        cka_values = []
        for m in common_models:
            val = cka_pretrained.get(m, {}).get(metric_key)
            if val is not None:
                cka_values.append(val)
            else:
                cka_values.append(0)

        for bench_name in BENCHMARK_DISPLAY.keys():
            bench_values = []
            valid = True
            for m in common_models:
                val = benchmark_results.get(m, {}).get(bench_name)
                if val is not None:
                    bench_values.append(val)
                else:
                    valid = False
                    break

            if not valid or len(bench_values) < 2:
                continue

            try:
                pearson_r, pearson_p = stats.pearsonr(cka_values, bench_values)
                spearman_r, spearman_p = stats.spearmanr(cka_values, bench_values)
            except Exception:
                continue

            key = f"pretrained_{metric_key}_vs_{bench_name}"
            correlations[key] = {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "n": len(common_models),
                "cka_type": f"pretrained_{metric_key}",
                "benchmark": bench_name,
            }

    # Post-projector CKA vs benchmarks
    common_internal = sorted(
        set(benchmark_results.keys()) &
        set(cka_internal.keys())
    )

    if len(common_internal) >= 2:
        for pair_label in ["CKA(VisionRaw, TextEmb)", "CKA(VisionProj, TextEmb)"]:
            cka_values = []
            for m in common_internal:
                val = cka_internal.get(m, {}).get(pair_label, {}).get("cka_linear")
                cka_values.append(val if val is not None else 0)

            for bench_name in BENCHMARK_DISPLAY.keys():
                bench_values = []
                valid = True
                for m in common_internal:
                    val = benchmark_results.get(m, {}).get(bench_name)
                    if val is not None:
                        bench_values.append(val)
                    else:
                        valid = False
                        break

                if not valid or len(bench_values) < 2:
                    continue

                try:
                    pearson_r, pearson_p = stats.pearsonr(cka_values, bench_values)
                    spearman_r, spearman_p = stats.spearmanr(cka_values, bench_values)
                except Exception:
                    continue

                safe_label = pair_label.replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")
                key = f"internal_{safe_label}_vs_{bench_name}"
                correlations[key] = {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                    "n": len(common_internal),
                    "cka_type": f"internal_{pair_label}",
                    "benchmark": bench_name,
                }

    return correlations


# ============================================================================
# Visualization
# ============================================================================
def plot_benchmark_heatmap(benchmark_results: dict, output_dir: str):
    """Plot benchmark scores as a heatmap."""
    models = [m for m in MODEL_ORDER if m in benchmark_results]
    benchmarks = [b for b in BENCHMARK_DISPLAY.keys()
                  if any(b in benchmark_results.get(m, {}) for m in models)]

    if not models or not benchmarks:
        print("  No data for benchmark heatmap")
        return

    data = np.zeros((len(models), len(benchmarks)))
    for i, m in enumerate(models):
        for j, b in enumerate(benchmarks):
            data[i, j] = benchmark_results.get(m, {}).get(b, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels([BENCHMARK_DISPLAY.get(b, b) for b in benchmarks], rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in models])

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(benchmarks)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmax(data) * 0.7 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                       color=text_color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Score")
    ax.set_title("Benchmark Scores by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_heatmap.png"), dpi=150)
    plt.close()
    print(f"  Saved benchmark_heatmap.png")


def plot_cka_vs_benchmark_scatter(benchmark_results: dict, cka_pretrained: dict,
                                  cka_internal: dict, output_dir: str):
    """Scatter plots: CKA (x) vs Benchmark Score (y), one subplot per benchmark."""
    benchmarks = [b for b in BENCHMARK_DISPLAY.keys()
                  if any(b in benchmark_results.get(m, {}) for m in MODEL_ORDER)]

    if not benchmarks:
        print("  No data for scatter plots")
        return

    n_benchmarks = len(benchmarks)
    fig, axes = plt.subplots(2, n_benchmarks, figsize=(5 * n_benchmarks, 10))
    if n_benchmarks == 1:
        axes = axes.reshape(2, 1)

    # Row 1: Pretrained CKA
    for j, bench in enumerate(benchmarks):
        ax = axes[0, j]
        for model_key in MODEL_ORDER:
            cka_val = cka_pretrained.get(model_key, {}).get("cka_linear")
            bench_val = benchmark_results.get(model_key, {}).get(bench)
            if cka_val is not None and bench_val is not None:
                ax.scatter(cka_val, bench_val,
                          color=MODEL_COLORS.get(model_key, "gray"),
                          marker=MODEL_MARKERS.get(model_key, "o"),
                          s=150, zorder=5, edgecolors="black", linewidth=0.5,
                          label=MODEL_DISPLAY.get(model_key, model_key))

        ax.set_xlabel("Pretrained CKA (linear)")
        ax.set_ylabel(BENCHMARK_DISPLAY.get(bench, bench))
        ax.set_title(f"Pretrained CKA vs {BENCHMARK_DISPLAY.get(bench, bench)}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="best")

    # Row 2: Post-Projector CKA
    for j, bench in enumerate(benchmarks):
        ax = axes[1, j]
        for model_key in MODEL_ORDER:
            internal = cka_internal.get(model_key, {})
            cka_val = internal.get("CKA(VisionProj, TextEmb)", {}).get("cka_linear")
            bench_val = benchmark_results.get(model_key, {}).get(bench)
            if cka_val is not None and bench_val is not None:
                ax.scatter(cka_val, bench_val,
                          color=MODEL_COLORS.get(model_key, "gray"),
                          marker=MODEL_MARKERS.get(model_key, "o"),
                          s=150, zorder=5, edgecolors="black", linewidth=0.5,
                          label=MODEL_DISPLAY.get(model_key, model_key))

        ax.set_xlabel("Post-Projector CKA (linear)")
        ax.set_ylabel(BENCHMARK_DISPLAY.get(bench, bench))
        ax.set_title(f"Post-Proj CKA vs {BENCHMARK_DISPLAY.get(bench, bench)}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="best")

    plt.suptitle("CKA vs Benchmark Performance", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cka_vs_benchmark_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cka_vs_benchmark_scatter.png")


def plot_cka_delta(cka_pretrained: dict, cka_internal: dict, output_dir: str):
    """Bar chart: Pretrained CKA vs Post-Projector CKA per model."""
    models = [m for m in MODEL_ORDER
              if m in cka_pretrained and m in cka_internal]

    if not models:
        print("  No data for CKA delta plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    pretrained_vals = []
    postproj_vals = []

    for m in models:
        pre = cka_pretrained.get(m, {}).get("cka_linear", 0)
        post_data = cka_internal.get(m, {}).get("CKA(VisionProj, TextEmb)", {})
        post = post_data.get("cka_linear", 0) if isinstance(post_data, dict) else 0
        pretrained_vals.append(pre)
        postproj_vals.append(post)

    bars1 = ax.bar(x - width/2, pretrained_vals, width, label="Pretrained CKA",
                   color="#4C8BF5", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, postproj_vals, width, label="Post-Projector CKA",
                   color="#E94B3C", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_ylabel("CKA (linear)")
    ax.set_title("Pretrained vs Post-Projector CKA", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(pretrained_vals, default=0), max(postproj_vals, default=0)) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cka_delta.png"), dpi=150)
    plt.close()
    print(f"  Saved cka_delta.png")


def plot_correlation_heatmap(correlations: dict, output_dir: str):
    """Heatmap of Pearson r between CKA types and benchmarks."""
    if not correlations:
        print("  No correlation data for heatmap")
        return

    # Organize by CKA type and benchmark
    cka_types = sorted(set(c["cka_type"] for c in correlations.values()))
    bench_names = sorted(set(c["benchmark"] for c in correlations.values()))

    if not cka_types or not bench_names:
        return

    data = np.full((len(cka_types), len(bench_names)), np.nan)
    for corr in correlations.values():
        i = cka_types.index(corr["cka_type"])
        j = bench_names.index(corr["benchmark"])
        data[i, j] = corr["pearson_r"]

    fig, ax = plt.subplots(figsize=(max(8, len(bench_names) * 1.5), max(4, len(cka_types) * 0.8)))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(bench_names)))
    ax.set_xticklabels([BENCHMARK_DISPLAY.get(b, b) for b in bench_names], rotation=45, ha="right")
    ax.set_yticks(range(len(cka_types)))
    ax.set_yticklabels(cka_types, fontsize=8)

    for i in range(len(cka_types)):
        for j in range(len(bench_names)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       color="white" if abs(val) > 0.5 else "black", fontsize=9)

    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("CKA-Benchmark Correlation (Pearson r)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"  Saved correlation_heatmap.png")


def plot_controlled_comparison(benchmark_results: dict, cka_pretrained: dict,
                               cka_internal: dict, output_dir: str):
    """Plot controlled comparison: Model 3 vs 4 (same LLM, different encoder)."""
    m3 = "llava_gemma_siglip2"
    m4 = "llava_gemma_clip"

    if m3 not in benchmark_results or m4 not in benchmark_results:
        print("  No data for controlled comparison (need both Gemma models)")
        return

    benchmarks = sorted(set(benchmark_results.get(m3, {}).keys()) &
                       set(benchmark_results.get(m4, {}).keys()))

    if not benchmarks:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Benchmark comparison
    ax = axes[0]
    x = np.arange(len(benchmarks))
    width = 0.35
    vals3 = [benchmark_results[m3].get(b, 0) for b in benchmarks]
    vals4 = [benchmark_results[m4].get(b, 0) for b in benchmarks]
    ax.bar(x - width/2, vals3, width, label="SigLIP2", color="#0ABAB5")
    ax.bar(x + width/2, vals4, width, label="CLIP", color="#E94B3C")
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_DISPLAY.get(b, b) for b in benchmarks], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Benchmark: SigLIP2 vs CLIP\n(Same LLM: Gemma3-4B)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: CKA comparison
    ax = axes[1]
    cka_types = ["Pretrained CKA", "Post-Proj CKA"]
    cka3 = [
        cka_pretrained.get(m3, {}).get("cka_linear", 0),
        cka_internal.get(m3, {}).get("CKA(VisionProj, TextEmb)", {}).get("cka_linear", 0)
        if isinstance(cka_internal.get(m3, {}).get("CKA(VisionProj, TextEmb)", {}), dict) else 0,
    ]
    cka4 = [
        cka_pretrained.get(m4, {}).get("cka_linear", 0),
        cka_internal.get(m4, {}).get("CKA(VisionProj, TextEmb)", {}).get("cka_linear", 0)
        if isinstance(cka_internal.get(m4, {}).get("CKA(VisionProj, TextEmb)", {}), dict) else 0,
    ]
    x2 = np.arange(len(cka_types))
    ax.bar(x2 - width/2, cka3, width, label="SigLIP2", color="#0ABAB5")
    ax.bar(x2 + width/2, cka4, width, label="CLIP", color="#E94B3C")
    ax.set_xticks(x2)
    ax.set_xticklabels(cka_types)
    ax.set_ylabel("CKA (linear)")
    ax.set_title("CKA: SigLIP2 vs CLIP")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Summary text
    ax = axes[2]
    ax.axis("off")
    summary_lines = [
        "Controlled Comparison",
        "=" * 30,
        f"LLM: Gemma3-4B (shared)",
        f"Encoder A: SigLIP2-SO400M",
        f"Encoder B: CLIP-ViT-L/14",
        "",
        "Key Question:",
        "Does higher CKA predict",
        "better benchmark scores?",
        "",
    ]

    # Add comparison data
    for b in benchmarks[:4]:
        v3 = benchmark_results[m3].get(b, 0)
        v4 = benchmark_results[m4].get(b, 0)
        winner = "SigLIP2" if v3 > v4 else "CLIP"
        summary_lines.append(f"{BENCHMARK_DISPLAY.get(b, b)}: {winner} wins")

    ax.text(0.1, 0.9, "\n".join(summary_lines), transform=ax.transAxes,
           fontsize=11, verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Controlled Comparison: Same LLM, Different Encoder",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "controlled_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved controlled_comparison.png")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze CKA vs benchmark performance")
    parser.add_argument("--benchmark_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "benchmark"))
    parser.add_argument("--cka_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "cka"))
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "analysis"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CKA vs Benchmark Performance Analysis")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading benchmark results...")
    benchmark_results = load_benchmark_results(args.benchmark_dir)

    print("\n[2/5] Loading CKA results...")
    cka_pretrained, cka_internal = load_cka_results(args.cka_dir)

    if not benchmark_results:
        print("\nNo benchmark results found. Run benchmarks first:")
        print("  bash scripts/run_benchmarks.sh")
        print("\nWill still analyze CKA data if available.")

    # Print data summary tables
    print(f"\n{'='*60}")
    print("BENCHMARK SCORES")
    print(f"{'='*60}")
    if benchmark_results:
        all_benchmarks = sorted(set(
            b for m in benchmark_results.values() for b in m.keys()
        ))
        header = f"{'Model':<25}" + "".join(f"{BENCHMARK_DISPLAY.get(b, b):>10}" for b in all_benchmarks)
        print(header)
        print("-" * len(header))
        for model_key in MODEL_ORDER:
            if model_key in benchmark_results:
                line = f"{MODEL_DISPLAY.get(model_key, model_key):<25}"
                for b in all_benchmarks:
                    val = benchmark_results[model_key].get(b)
                    line += f"{val:>10.1f}" if val is not None else f"{'N/A':>10}"
                print(line)

    print(f"\n{'='*60}")
    print("PRETRAINED CKA SCORES")
    print(f"{'='*60}")
    if cka_pretrained:
        print(f"{'Model':<25} {'CKA(lin)':>10} {'CKA(rbf)':>10}")
        print("-" * 45)
        for model_key in MODEL_ORDER:
            if model_key in cka_pretrained:
                m = cka_pretrained[model_key]
                name = MODEL_DISPLAY.get(model_key, model_key)
                print(f"{name:<25} {m.get('cka_linear', 0):>10.4f} {m.get('cka_rbf', 0):>10.4f}")

    print(f"\n{'='*60}")
    print("INTERNAL CKA SCORES")
    print(f"{'='*60}")
    if cka_internal:
        print(f"{'Model':<25} {'PreProj':>10} {'PostProj':>10} {'Delta':>8}")
        print("-" * 55)
        for model_key in MODEL_ORDER:
            if model_key in cka_internal:
                res = cka_internal[model_key]
                name = MODEL_DISPLAY.get(model_key, model_key)
                pre = res.get("CKA(VisionRaw, TextEmb)", {})
                post = res.get("CKA(VisionProj, TextEmb)", {})
                pre_val = pre.get("cka_linear") if isinstance(pre, dict) else None
                post_val = post.get("cka_linear") if isinstance(post, dict) else None
                pre_str = f"{pre_val:.4f}" if pre_val is not None else "N/A"
                post_str = f"{post_val:.4f}" if post_val is not None else "N/A"
                delta = f"{post_val - pre_val:+.4f}" if pre_val and post_val else "N/A"
                print(f"{name:<25} {pre_str:>10} {post_str:>10} {delta:>8}")

    # Correlation analysis
    print(f"\n[3/5] Computing correlations...")
    correlations = compute_correlations(benchmark_results, cka_pretrained, cka_internal)

    if correlations:
        print(f"\n{'='*60}")
        print("CORRELATION ANALYSIS (CKA vs Benchmark)")
        print(f"{'='*60}")
        print(f"{'CKA Type':<40} {'Benchmark':<15} {'Pearson r':>10} {'p':>8} {'n':>4}")
        print("-" * 80)
        for key, corr in sorted(correlations.items()):
            bench_display = BENCHMARK_DISPLAY.get(corr["benchmark"], corr["benchmark"])
            sig = "*" if corr["pearson_p"] < 0.05 else ""
            print(f"{corr['cka_type']:<40} {bench_display:<15} "
                  f"{corr['pearson_r']:>9.3f}{sig} {corr['pearson_p']:>7.3f} {corr['n']:>4}")

        # Overall summary
        r_values = [c["pearson_r"] for c in correlations.values()]
        mean_r = np.mean(r_values)
        print(f"\nMean |r|: {np.mean(np.abs(r_values)):.3f}")
        print(f"Mean r:   {mean_r:.3f}")
        if mean_r > 0.5:
            print("=> POSITIVE correlation: Higher CKA tends to predict better performance")
        elif mean_r < -0.5:
            print("=> NEGATIVE correlation: Higher CKA does NOT predict better performance (Paradox!)")
        else:
            print("=> WEAK correlation: CKA is not a strong predictor of performance")

    # Visualizations
    print(f"\n[4/5] Generating visualizations...")
    if benchmark_results:
        plot_benchmark_heatmap(benchmark_results, args.output_dir)
    if cka_pretrained or cka_internal:
        plot_cka_vs_benchmark_scatter(benchmark_results, cka_pretrained, cka_internal, args.output_dir)
        plot_cka_delta(cka_pretrained, cka_internal, args.output_dir)
        plot_correlation_heatmap(correlations, args.output_dir)
        plot_controlled_comparison(benchmark_results, cka_pretrained, cka_internal, args.output_dir)

    # Save all results
    print(f"\n[5/5] Saving results...")
    summary = {
        "benchmark_results": benchmark_results,
        "pretrained_cka": {k: {kk: vv for kk, vv in v.items()
                               if not isinstance(vv, np.ndarray)}
                          for k, v in cka_pretrained.items()},
        "internal_cka": cka_internal,
        "correlations": correlations,
        "n_models": len(set(benchmark_results.keys()) |
                       set(cka_pretrained.keys()) |
                       set(cka_internal.keys())),
    }
    with open(os.path.join(args.output_dir, "full_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll results saved to {args.output_dir}/")
    print("  - full_analysis.json")
    print("  - benchmark_heatmap.png")
    print("  - cka_vs_benchmark_scatter.png")
    print("  - cka_delta.png")
    print("  - correlation_heatmap.png")
    print("  - controlled_comparison.png")


if __name__ == "__main__":
    main()
