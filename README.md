# VLM Encoder Alignment Toolkit

Analyze how different vision encoders (CLIP, SigLIP, DINOv2, InternViT, PaliGemma) align with Large Language Models through visual analysis tools.

**Key Features:**
- **Inference Speed Benchmark** - Latency, throughput, and memory profiling for all encoders
- **Attention Map Visualization** - Patch-text similarity heatmaps showing where models focus
- **CKA Alignment Analysis** - Centered Kernel Alignment between vision and text embeddings
- **Deep CORAL Analysis** - Covariance alignment for text-image, text-text, image-image comparison
- **EAS (Enhanced Alignment Score)** - CKA + CORAL + Discriminability combined metric
- **Embedding Space Visualization** - t-SNE/UMAP projections with Procrustes alignment
- **ELAS Score** - Composite Encoder-LLM Alignment Score for optimal pairing
- **E2E Validation** - CKA vs actual retrieval performance (CKA-Performance Paradox)

Supports both **CLI** and **Gradio Web UI**.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yujuyeon0511/vlm-encoder-alignment.git
cd vlm-encoder-alignment
pip install -r requirements.txt
```

### Run with Sample Data (no external dataset needed)

```bash
# Encoder comparison with built-in sample data
python cli.py compare --n-samples 3

# Inference speed benchmark
python cli.py speed --encoders clip siglip dinov2

# Launch web UI
python app.py
```

### Run with Custom Data

```bash
# Set data path via environment variable
export VLM_DATA_ROOT=/path/to/your/vlm_data
python cli.py compare --n-samples 30

# Or pass directly
python cli.py compare --data-root /path/to/your/vlm_data
```

---

## CLI Reference

```
python cli.py <command> [options]
```

| Command | Description |
|---------|-------------|
| `compare` | Encoder comparison (CKA scores, projector evaluation) |
| `coral` | Deep CORAL analysis (CKA + CORAL + EAS) |
| `multi-llm` | Compare encoders across multiple LLMs |
| `speed` | Inference speed benchmark with dashboard |
| `attention` | Attention map visualization (CLIP, SigLIP) |
| `embedding` | t-SNE/UMAP embedding space visualization |
| `elas` | ELAS (Encoder-LLM Alignment Score) calculation |
| `e2e` | End-to-end validation (CKA vs task performance) |
| `all` | Run compare + speed + embedding |

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--encoders` | clip siglip dinov2 | Vision encoders to use |
| `--llms` | llama | Target LLMs |
| `--n-samples` | 30 | Samples per data type |
| `--data-root` | (auto) | Path to real VLM dataset |
| `--output-dir` | outputs | Output directory for plots |
| `--device` | auto | cuda or cpu |

### Examples

```bash
# Speed benchmark with specific batch sizes
python cli.py speed --encoders clip siglip --batch-sizes 1 4 8 16 32

# Attention maps for a specific image
python cli.py attention --image path/to/image.png --text "What is shown?"

# Deep CORAL analysis (text-image alignment comparison)
python cli.py coral --encoders clip siglip dinov2 --llms llama

# ELAS across multiple LLMs
python cli.py elas --encoders clip siglip dinov2 --llms llama qwen

# Embedding visualization with UMAP
python cli.py embedding --method umap --n-samples 50
```

---

## Gradio Web UI

```bash
python app.py
# Opens at http://localhost:7860
```

**6 Tabs:**
1. **Inference Speed** - Select encoders, run benchmark, view dashboard
2. **Attention Maps** - Upload image, see attention heatmaps
3. **Alignment Analysis** - CKA/MSE/Cosine metrics per encoder
4. **Embedding Space** - Interactive t-SNE/UMAP scatter plots
5. **CORAL Analysis** - Deep CORAL covariance alignment + EAS dashboard
6. **E2E Validation** - CKA-Performance Paradox detection

---

## Configuration

Edit `config.yaml` to customize model IDs and data paths:

```yaml
data:
  root: /path/to/your/VLM_DATA  # or set VLM_DATA_ROOT env var

models:
  vision_encoders:
    clip: openai/clip-vit-base-patch32
    siglip: google/siglip-base-patch16-224
    dinov2: facebook/dinov2-base
  llms:
    llama: huggyllama/llama-7b
    qwen: Qwen/Qwen2.5-7B
```

---

## Project Structure

```
vlm-encoder-alignment/
├── cli.py                       # Unified CLI
├── app.py                       # Gradio web UI
├── config.yaml                  # Central configuration
├── sample_data/                 # Built-in sample images
│
├── vlm_alignment/               # Main package
│   ├── config.py                # Config loader
│   ├── models/
│   │   ├── vision_encoders.py   # All vision encoder loading + embedding extraction
│   │   ├── llm_loaders.py       # All LLM loading + text embedding extraction
│   │   └── projectors.py        # Projector architectures + training
│   ├── analysis/
│   │   ├── cka.py               # CKA implementation
│   │   ├── coral.py             # Deep CORAL analysis + EAS
│   │   ├── alignment.py         # Alignment analysis + ELAS
│   │   └── speed_benchmark.py   # Inference speed profiling
│   ├── visualization/
│   │   ├── plot_style.py        # Unified plot styling
│   │   ├── attention_maps.py    # Attention heatmap overlays
│   │   ├── embedding_space.py   # t-SNE/UMAP/Procrustes plots
│   │   ├── alignment_plots.py   # CKA heatmaps, comparison charts
│   │   ├── coral_plots.py       # CORAL/EAS dashboard plots
│   │   └── speed_plots.py       # Speed benchmark dashboard
│   ├── data/
│   │   ├── synthetic.py         # Synthetic data generation
│   │   └── dataset.py           # Real dataset loader (configurable path)
│   └── experiments/
│       ├── encoder_comparison.py
│       ├── coral_alignment.py   # CORAL experiment runner
│       ├── e2e_validation.py
│       ├── elas_score.py
│       └── speed_benchmark.py
│
└── scripts/
    ├── generate_sample_data.py
    └── generate_paper_figures.py
```

---

## CKA vs Deep CORAL

| Metric | What it measures | Limitation |
|--------|-----------------|------------|
| **CKA** | Structural similarity (kernel alignment) | Ignores distribution shape; high CKA =/= good task performance |
| **CORAL** | Second-order statistics (covariance alignment) | Captures distributional spread and feature correlations |
| **EAS** | CKA + CORAL + Discriminability | Balanced metric that addresses CKA-Performance Paradox |

**Why CORAL?** CKA measures whether two spaces have _similar patterns_, but ignores _how features are distributed_. CORAL directly aligns covariance matrices, giving a more actionable signal for text-image, text-text, and image-image alignment.

```
EAS = 0.3 * CKA + 0.3 * CORAL_similarity + 0.4 * Discriminability
```

---

## Key Research Findings

| Finding | Detail |
|---------|--------|
| **SigLIP superiority** | +9.4% higher CKA vs CLIP overall |
| **Task-adaptive selection** | SigLIP best for charts, DINOv2 best for tables |
| **LLM-specific pairing** | DINOv2 best for LLaMA, SigLIP best for Qwen |
| **CKA-Performance Paradox** | Higher CKA correlates _negatively_ with retrieval (r=-0.99) |

---

## Supported Models

### Vision Encoders
| Name | Model ID | Dim |
|------|----------|-----|
| CLIP | openai/clip-vit-base-patch32 | 768 |
| SigLIP | google/siglip-base-patch16-224 | 768 |
| DINOv2 | facebook/dinov2-base | 768 |
| InternViT | OpenGVLab/InternViT-300M-448px | 768 |
| PaliGemma | google/paligemma-3b-pt-224 | 2048 |

### LLMs
| Name | Model ID |
|------|----------|
| LLaMA | huggyllama/llama-7b |
| LLaMA 3.1 | meta-llama/Llama-3.1-8B |
| Qwen 2.5 | Qwen/Qwen2.5-7B |
| Gemma 3 | google/gemma-3-4b-pt |
| InternLM 2.5 | internlm/internlm2_5-7b |

---

## License

MIT
