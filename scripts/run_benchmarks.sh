#!/bin/bash
# =============================================================================
# Run lmms-eval benchmarks for CKA vs Performance validation
# Usage: bash scripts/run_benchmarks.sh
# Environment: docmllm conda env, 2x A100 40GB
# =============================================================================
set -e

CONDA_ENV="docmllm"
OUTPUT_BASE="/NetDisk/juyeon/vlm-encoder-alignment/outputs/benchmark"
TASKS="textvqa_val,chartqa,docvqa_val,mmbench_en_dev,mmstar,pope"
LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "VLM Benchmark Evaluation"
echo "Tasks: ${TASKS}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================"

# -------------------------------------------
# Model 1: Qwen2.5-VL-7B-Instruct
# -------------------------------------------
echo "[1/4] Qwen2.5-VL-7B-Instruct"
conda run -n ${CONDA_ENV} python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=/NetDisk/juyeon/models/Qwen2.5-VL-7B-Instruct,device_map=auto \
    --tasks ${TASKS} \
    --batch_size 4 \
    --log_samples \
    --output_path ${OUTPUT_BASE}/qwen25vl \
    2>&1 | tee ${LOG_DIR}/qwen25vl.log

echo "[1/4] Qwen2.5-VL-7B done."

# -------------------------------------------
# Model 2: LLaVA-OneVision-Qwen2-7B
# -------------------------------------------
echo "[2/4] LLaVA-OneVision-Qwen2-7B"
conda run -n ${CONDA_ENV} python -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/NetDisk/juyeon/AdaMMS/llava-onevision-qwen2-7b-si,device_map=auto,conv_template=qwen_1_5 \
    --tasks ${TASKS} \
    --batch_size 4 \
    --log_samples \
    --output_path ${OUTPUT_BASE}/llava_ov \
    2>&1 | tee ${LOG_DIR}/llava_ov.log

echo "[2/4] LLaVA-OneVision done."

# -------------------------------------------
# Model 3: LLaVA-Gemma-SigLIP2 (LoRA)
# Model 4: LLaVA-Gemma-CLIP (LoRA)
# These use custom LLaVA-Gemma architecture with LoRA adapters.
# We use a dedicated Python script to handle loading and evaluation.
# -------------------------------------------
echo "[3/4] LLaVA-Gemma-SigLIP2 (LoRA)"
conda run -n ${CONDA_ENV} python /NetDisk/juyeon/vlm-encoder-alignment/scripts/run_lmms_eval_llava_gemma.py \
    --model_path /NetDisk/juyeon/llava-sp/two_stage_koni_siglip2_4n_251107_0036_stage2_finetune \
    --model_name llava_gemma_siglip2 \
    --tasks ${TASKS} \
    --batch_size 4 \
    --output_path ${OUTPUT_BASE}/llava_gemma_siglip2 \
    2>&1 | tee ${LOG_DIR}/llava_gemma_siglip2.log

echo "[3/4] LLaVA-Gemma-SigLIP2 done."

echo "[4/4] LLaVA-Gemma-CLIP (LoRA)"
conda run -n ${CONDA_ENV} python /NetDisk/juyeon/vlm-encoder-alignment/scripts/run_lmms_eval_llava_gemma.py \
    --model_path /NetDisk/juyeon/llava-sp/4n_llava-gemma3-4b-kisti-finetune_lora_testtt_250917 \
    --model_name llava_gemma_clip \
    --tasks ${TASKS} \
    --batch_size 4 \
    --output_path ${OUTPUT_BASE}/llava_gemma_clip \
    2>&1 | tee ${LOG_DIR}/llava_gemma_clip.log

echo "[4/4] LLaVA-Gemma-CLIP done."

echo "========================================"
echo "All benchmarks complete!"
echo "Results in: ${OUTPUT_BASE}"
echo "========================================"
