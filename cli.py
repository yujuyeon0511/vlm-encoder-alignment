#!/usr/bin/env python3
"""Unified CLI for VLM Encoder Alignment Toolkit.

Usage:
    python cli.py compare       # Encoder comparison (CKA, projectors)
    python cli.py coral         # CORAL alignment analysis (CKA + CORAL + EAS)
    python cli.py speed         # Inference speed benchmark
    python cli.py attention     # Attention map visualization
    python cli.py embedding     # Embedding space visualization
    python cli.py elas          # ELAS score calculation
    python cli.py e2e           # End-to-end validation
    python cli.py all           # Run all experiments

Common options:
    --encoders clip siglip dinov2
    --llms llama qwen
    --data-root /path/to/data
    --output-dir outputs
    --device cuda
    --n-samples 30
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_compare(args):
    """Run encoder comparison experiment."""
    from vlm_alignment.experiments.encoder_comparison import run_encoder_comparison
    run_encoder_comparison(
        encoder_names=args.encoders,
        llm_name=args.llms[0] if args.llms else "llama",
        n_samples=args.n_samples,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
    )


def cmd_multi_llm(args):
    """Run multi-LLM comparison."""
    from vlm_alignment.experiments.encoder_comparison import run_multi_llm_comparison
    run_multi_llm_comparison(
        encoder_names=args.encoders,
        llm_names=args.llms,
        n_samples=args.n_samples,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
    )


def cmd_speed(args):
    """Run speed benchmark."""
    from vlm_alignment.experiments.speed_benchmark import run_speed_benchmark
    batch_sizes = [int(b) for b in args.batch_sizes] if args.batch_sizes else None
    run_speed_benchmark(
        encoder_names=args.encoders,
        llm_names=args.llms if args.llms else None,
        batch_sizes=batch_sizes,
        output_dir=args.output_dir,
        device=args.device,
    )


def cmd_attention(args):
    """Generate attention map visualizations."""
    from vlm_alignment.visualization.attention_maps import plot_attention_comparison
    from vlm_alignment.data.dataset import VLMDataset
    from PIL import Image

    if args.image:
        image = Image.open(args.image).convert("RGB")
        text = args.text or "Describe this image."
    else:
        dataset = VLMDataset(data_root=args.data_root)
        samples = dataset.load_samples("chart", n_samples=1)
        if samples:
            image, text = samples[0].image, samples[0].question
        else:
            from vlm_alignment.data.synthetic import DataGenerator
            gen = DataGenerator()
            image = gen.generate_bar_chart()
            text = "What is the highest value in this chart?"

    encoders = args.encoders or ["clip", "siglip"]
    # Only clip and siglip support patch-text similarity
    encoders = [e for e in encoders if e in ("clip", "siglip")]
    if not encoders:
        encoders = ["clip", "siglip"]

    os.makedirs(args.output_dir, exist_ok=True)
    plot_attention_comparison(
        image, text,
        encoder_names=encoders,
        output_path=os.path.join(args.output_dir, "attention_comparison.png"),
        device=args.device or "cuda",
    )


def cmd_embedding(args):
    """Generate embedding space visualizations."""
    from vlm_alignment.models.vision_encoders import VisionEncoderManager
    from vlm_alignment.data.dataset import VLMDataset
    from vlm_alignment.visualization.embedding_space import (
        plot_multi_encoder_tsne, plot_similarity_matrices,
    )

    dataset = VLMDataset(data_root=args.data_root)
    samples, labels = dataset.load_mixed(n_per_type=args.n_samples)
    images, _ = dataset.get_images_and_texts(samples)

    if not images:
        print("No data available.")
        return

    device = args.device or "cuda"
    mgr = VisionEncoderManager(device=device)
    encoders = args.encoders or ["clip", "siglip", "dinov2"]
    embs = mgr.extract_multi_encoder(encoders, images)
    mgr.unload()

    os.makedirs(args.output_dir, exist_ok=True)
    method = args.method if hasattr(args, "method") and args.method else "tsne"
    plot_multi_encoder_tsne(embs, labels=labels, method=method,
                            output_path=os.path.join(args.output_dir, f"embedding_{method}.png"))
    plot_similarity_matrices(embs, output_path=os.path.join(args.output_dir, "similarity_matrices.png"))


def cmd_elas(args):
    """Run ELAS experiment."""
    from vlm_alignment.experiments.elas_score import run_elas_experiment
    run_elas_experiment(
        encoder_names=args.encoders,
        llm_names=args.llms,
        n_samples=args.n_samples,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
    )


def cmd_coral(args):
    """Run CORAL alignment analysis."""
    from vlm_alignment.experiments.coral_alignment import run_coral_analysis
    run_coral_analysis(
        encoder_names=args.encoders,
        llm_name=args.llms[0] if args.llms else "llama",
        n_samples=args.n_samples,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
    )


def cmd_e2e(args):
    """Run E2E validation."""
    from vlm_alignment.experiments.e2e_validation import run_e2e_validation
    run_e2e_validation(
        encoder_names=args.encoders,
        llm_name=args.llms[0] if args.llms else "llama",
        n_samples=args.n_samples,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
    )


def cmd_all(args):
    """Run all experiments."""
    print("Running all experiments...\n")
    for cmd in [cmd_compare, cmd_speed, cmd_embedding]:
        try:
            cmd(args)
        except Exception as e:
            print(f"Warning: {cmd.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
        print()


def main():
    parser = argparse.ArgumentParser(
        description="VLM Encoder Alignment Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Shared arguments
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--encoders", nargs="+", default=None,
                        help="Vision encoders (e.g., clip siglip dinov2)")
    parent.add_argument("--llms", nargs="+", default=None,
                        help="Target LLMs (e.g., llama qwen)")
    parent.add_argument("--data-root", type=str, default=None,
                        help="Path to real VLM dataset")
    parent.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    parent.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu (default: auto)")
    parent.add_argument("--n-samples", type=int, default=30,
                        help="Samples per data type (default: 30)")

    # Subcommands
    sub.add_parser("compare", parents=[parent], help="Encoder comparison (CKA, projectors)")
    sub.add_parser("multi-llm", parents=[parent], help="Multi-LLM comparison")
    sub.add_parser("coral", parents=[parent], help="CORAL alignment analysis (CKA + CORAL + EAS)")
    sub.add_parser("elas", parents=[parent], help="ELAS score calculation")
    sub.add_parser("e2e", parents=[parent], help="End-to-end validation")
    sub.add_parser("all", parents=[parent], help="Run all experiments")

    speed_p = sub.add_parser("speed", parents=[parent], help="Inference speed benchmark")
    speed_p.add_argument("--batch-sizes", nargs="+", default=None,
                         help="Batch sizes to test (e.g., 1 4 8 16)")

    att_p = sub.add_parser("attention", parents=[parent], help="Attention map visualization")
    att_p.add_argument("--image", type=str, default=None, help="Path to input image")
    att_p.add_argument("--text", type=str, default=None, help="Query text")

    emb_p = sub.add_parser("embedding", parents=[parent], help="Embedding space visualization")
    emb_p.add_argument("--method", choices=["tsne", "umap"], default="tsne",
                        help="Dimension reduction method")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\nRun 'python cli.py <command> --help' for details on each command.")
        return

    import torch
    print("=" * 60)
    print("VLM Encoder Alignment Toolkit")
    print("=" * 60)
    print(f"Command: {args.command}")
    print(f"Device: {args.device or ('CUDA' if torch.cuda.is_available() else 'CPU')}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    commands = {
        "compare": cmd_compare,
        "multi-llm": cmd_multi_llm,
        "speed": cmd_speed,
        "attention": cmd_attention,
        "embedding": cmd_embedding,
        "coral": cmd_coral,
        "elas": cmd_elas,
        "e2e": cmd_e2e,
        "all": cmd_all,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
