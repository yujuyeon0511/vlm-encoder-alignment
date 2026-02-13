"""Inference speed benchmarking for vision encoders and LLMs.

NEW module: Measures latency, throughput, and memory usage for all
model components in the VLM pipeline.
"""

import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from vlm_alignment.config import get_device


@dataclass
class SpeedResult:
    """Benchmark result for a single model configuration."""

    model_name: str
    component: str  # 'vision_encoder', 'llm', 'projector'
    batch_size: int
    latency_ms: float  # Average latency per batch
    throughput: float  # Items per second
    memory_mb: float  # Peak GPU memory in MB
    std_ms: float = 0.0  # Standard deviation of latency


@dataclass
class FullBenchmarkResult:
    """Complete benchmark results across models and batch sizes."""

    results: List[SpeedResult] = field(default_factory=list)

    def summary_table(self) -> str:
        """Return a formatted summary table."""
        lines = [
            f"{'Model':<15} {'Component':<17} {'Batch':<6} "
            f"{'Latency(ms)':<13} {'Throughput':<13} {'Memory(MB)':<10}"
        ]
        lines.append("-" * 80)
        for r in sorted(self.results, key=lambda x: (x.component, x.model_name, x.batch_size)):
            lines.append(
                f"{r.model_name:<15} {r.component:<17} {r.batch_size:<6} "
                f"{r.latency_ms:<13.2f} {r.throughput:<13.1f} {r.memory_mb:<10.1f}"
            )
        return "\n".join(lines)


class InferenceSpeedBenchmark:
    """Benchmark inference speed of VLM components.

    Measures:
    - Vision encoder forward pass latency and throughput
    - LLM text embedding extraction speed
    - Projector transformation speed
    - GPU memory consumption
    """

    def __init__(
        self,
        device: Optional[str] = None,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
    ):
        self.device = device or get_device()
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def _get_gpu_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0

    def _generate_dummy_images(self, n: int, size: int = 224) -> List[Image.Image]:
        """Generate random images for benchmarking."""
        return [
            Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
            for _ in range(n)
        ]

    def _generate_dummy_texts(self, n: int) -> List[str]:
        """Generate dummy texts for benchmarking."""
        templates = [
            "What is shown in this image?",
            "Describe the chart data values.",
            "Read the table contents.",
            "What type of document is this?",
            "Summarize the information presented.",
        ]
        return [templates[i % len(templates)] for i in range(n)]

    def benchmark_vision_encoder(
        self,
        encoder_name: str,
        batch_sizes: List[int] = None,
        image_size: int = 224,
    ) -> List[SpeedResult]:
        """Benchmark a vision encoder across batch sizes.

        Args:
            encoder_name: e.g., 'clip', 'siglip', 'dinov2'
            batch_sizes: List of batch sizes to test
            image_size: Input image resolution

        Returns:
            List of SpeedResult for each batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        from vlm_alignment.models.vision_encoders import VisionEncoderManager

        mgr = VisionEncoderManager(device=self.device)
        model, processor = mgr.load(encoder_name)

        results = []
        for bs in batch_sizes:
            images = self._generate_dummy_images(bs, image_size)

            # Prepare inputs
            if encoder_name in ("clip", "siglip"):
                inputs = processor(images=images, return_tensors="pt", padding=True)
            else:
                inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Warmup
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(self.warmup_runs):
                    if encoder_name == "clip":
                        model.get_image_features(**inputs)
                    elif encoder_name == "siglip":
                        model.get_image_features(**inputs)
                    else:
                        model(**inputs)

            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            latencies = []
            for _ in range(self.benchmark_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.no_grad():
                    if encoder_name in ("clip", "siglip"):
                        model.get_image_features(**inputs)
                    else:
                        model(**inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

            avg_ms = np.mean(latencies)
            std_ms = np.std(latencies)
            memory_mb = self._get_gpu_memory_mb()
            throughput = bs / (avg_ms / 1000)

            results.append(
                SpeedResult(
                    model_name=encoder_name,
                    component="vision_encoder",
                    batch_size=bs,
                    latency_ms=avg_ms,
                    throughput=throughput,
                    memory_mb=memory_mb,
                    std_ms=std_ms,
                )
            )
            print(
                f"  {encoder_name} bs={bs}: {avg_ms:.1f}ms "
                f"({throughput:.1f} img/s, {memory_mb:.0f}MB)"
            )

        mgr.unload(encoder_name)
        return results

    def benchmark_llm(
        self,
        llm_name: str,
        batch_sizes: List[int] = None,
        max_length: int = 128,
    ) -> List[SpeedResult]:
        """Benchmark LLM text embedding extraction.

        Args:
            llm_name: e.g., 'llama', 'qwen', 'gemma3'
            batch_sizes: List of batch sizes to test
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]

        from vlm_alignment.models.llm_loaders import LLMManager

        mgr = LLMManager(device=self.device)
        model, tokenizer = mgr.load(llm_name)

        results = []
        for bs in batch_sizes:
            texts = self._generate_dummy_texts(bs)
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Warmup
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(self.warmup_runs):
                    model(**inputs)

            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            latencies = []
            for _ in range(self.benchmark_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    model(**inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

            avg_ms = np.mean(latencies)
            std_ms = np.std(latencies)
            memory_mb = self._get_gpu_memory_mb()
            throughput = bs / (avg_ms / 1000)

            results.append(
                SpeedResult(
                    model_name=llm_name,
                    component="llm",
                    batch_size=bs,
                    latency_ms=avg_ms,
                    throughput=throughput,
                    memory_mb=memory_mb,
                    std_ms=std_ms,
                )
            )
            print(
                f"  {llm_name} bs={bs}: {avg_ms:.1f}ms "
                f"({throughput:.1f} text/s, {memory_mb:.0f}MB)"
            )

        mgr.unload(llm_name)
        return results

    def benchmark_projector(
        self,
        input_dim: int = 768,
        output_dim: int = 4096,
        projection_types: List[str] = None,
        batch_sizes: List[int] = None,
    ) -> List[SpeedResult]:
        """Benchmark projector forward pass speed.

        Args:
            input_dim: Vision encoder output dimension
            output_dim: LLM hidden dimension
            projection_types: List of projector types to test
            batch_sizes: List of batch sizes
        """
        from vlm_alignment.models.projectors import create_projector

        if projection_types is None:
            projection_types = ["linear", "mlp", "2layer_mlp"]
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 128]

        results = []
        for proj_type in projection_types:
            proj = create_projector(proj_type, input_dim, output_dim).to(self.device)
            proj.eval()

            for bs in batch_sizes:
                dummy = torch.randn(bs, input_dim, device=self.device)

                # Warmup
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for _ in range(self.warmup_runs):
                        proj(dummy)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                latencies = []
                for _ in range(self.benchmark_runs * 5):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        proj(dummy)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000)

                avg_ms = np.mean(latencies)
                throughput = bs / (avg_ms / 1000)
                memory_mb = self._get_gpu_memory_mb()

                results.append(
                    SpeedResult(
                        model_name=proj_type,
                        component="projector",
                        batch_size=bs,
                        latency_ms=avg_ms,
                        throughput=throughput,
                        memory_mb=memory_mb,
                        std_ms=np.std(latencies),
                    )
                )

            del proj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def run_full_benchmark(
        self,
        encoder_names: List[str] = None,
        llm_names: List[str] = None,
        batch_sizes: List[int] = None,
    ) -> FullBenchmarkResult:
        """Run complete benchmark across all components.

        Args:
            encoder_names: Vision encoders to benchmark (default: clip, siglip, dinov2)
            llm_names: LLMs to benchmark (default: None = skip)
            batch_sizes: Batch sizes for vision encoders
        """
        if encoder_names is None:
            encoder_names = ["clip", "siglip", "dinov2"]

        full = FullBenchmarkResult()

        print("=" * 60)
        print("Vision Encoder Speed Benchmark")
        print("=" * 60)
        for name in encoder_names:
            print(f"\n[{name.upper()}]")
            full.results.extend(
                self.benchmark_vision_encoder(name, batch_sizes=batch_sizes)
            )

        if llm_names:
            print("\n" + "=" * 60)
            print("LLM Embedding Speed Benchmark")
            print("=" * 60)
            for name in llm_names:
                print(f"\n[{name.upper()}]")
                full.results.extend(self.benchmark_llm(name))

        print("\n" + "=" * 60)
        print("Projector Speed Benchmark")
        print("=" * 60)
        full.results.extend(self.benchmark_projector())

        print("\n" + "=" * 60)
        print("FULL BENCHMARK SUMMARY")
        print("=" * 60)
        print(full.summary_table())

        return full
