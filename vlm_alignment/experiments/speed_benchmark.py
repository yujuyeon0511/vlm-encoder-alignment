"""Speed benchmark experiment runner.

Runs the full inference speed benchmark and generates visualization dashboard.
"""

from typing import List, Optional, Dict
import os

from vlm_alignment.config import get_output_dir, get_device
from vlm_alignment.analysis.speed_benchmark import InferenceSpeedBenchmark
from vlm_alignment.visualization.speed_plots import (
    plot_latency_comparison,
    plot_throughput_by_batch,
    plot_memory_comparison,
    plot_full_benchmark_dashboard,
)


def run_speed_benchmark(
    encoder_names: List[str] = None,
    llm_names: List[str] = None,
    batch_sizes: List[int] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> Dict:
    """Run complete speed benchmark.

    Args:
        encoder_names: Vision encoders to benchmark (default: clip, siglip, dinov2)
        llm_names: LLMs to benchmark (default: None = skip LLM benchmark)
        batch_sizes: Batch sizes for vision encoder benchmark
        output_dir: Directory for output plots
        device: torch device
    """
    if encoder_names is None:
        encoder_names = ["clip", "siglip", "dinov2"]
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]
    if output_dir is None:
        output_dir = str(get_output_dir())
    if device is None:
        device = get_device()

    os.makedirs(output_dir, exist_ok=True)

    bench = InferenceSpeedBenchmark(
        device=device, warmup_runs=warmup_runs, benchmark_runs=benchmark_runs
    )

    result = bench.run_full_benchmark(
        encoder_names=encoder_names,
        llm_names=llm_names,
        batch_sizes=batch_sizes,
    )

    # Generate individual plots
    ve_results = [r for r in result.results if r.component == "vision_encoder"]

    plot_latency_comparison(
        ve_results, batch_size=1,
        output_path=os.path.join(output_dir, "speed_latency.png"),
    )
    plot_throughput_by_batch(
        ve_results,
        output_path=os.path.join(output_dir, "speed_throughput.png"),
    )
    plot_memory_comparison(
        ve_results, batch_size=1,
        output_path=os.path.join(output_dir, "speed_memory.png"),
    )

    # Full dashboard
    plot_full_benchmark_dashboard(
        result,
        output_path=os.path.join(output_dir, "speed_dashboard.png"),
    )

    return {"benchmark": result}
