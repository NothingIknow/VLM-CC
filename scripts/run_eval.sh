#!/bin/bash
# VLM-CC direction iteration eval (angle-based; defaults match iter_test_speed.sh)
# Edit paths/flags below, then: bash scripts/run_eval.sh

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 python scripts/eval/direction_iteration.py \
    --model_path models/nus-cube-inter_qwen2.5vl-7b \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --input_path assets/smoke_test \
    --output_dir outputs/eval \
    --iterations 30 \
    --start_angle 3 \
    --end_angle 0.1 \
    --angle_reduction_factor 0.5 \
    --img_size 256

# BASE_MODEL — pick one:
#   Qwen/Qwen2.5-VL-7B-Instruct          HuggingFace ID (auto-download to HF cache)
#   /path/to/Qwen2.5-VL-7B-Instruct      local model directory (offline)
#
# Optional flags (append to the command above):
#   --nogw                  (gray-world first step is ON by default; this disables it)
#   --camera canon5d
#   --infer_dtype bfloat16  (if your gpu memory is not enough)
#   --save_intermediate_images
#   --save_final_image
#
# Example --input_path:
#   assets/smoke_test
#   /path/to/gehler/test
#   /path/to/NUS-8/testset-200_PNG
#
# Paper LoRA checkpoints: models/nus-cube-inter_qwen2.5vl-7b | models/gehler-cube-inter_qwen2.5vl-7b | models/nus-gehler-inter_qwen2.5vl-7b
