#!/bin/bash
# Ablation study: Compare different image resolutions and timesteps
# Checkpoint: 0006840
# Dataset: AI2ThorPathTracing_10 (10 samples)

set -e

echo "========================================"
echo "Ablation Study: Image Resolution & Timesteps"
echo "Checkpoint: ckpt_pat/0006840"
echo "Dataset: AI2ThorPathTracing_10"
echo "========================================"

# Ablation 1: Default settings (1024x1024, 50 timesteps)
echo ""
echo "[1/2] Ablation 1: 1024x1024 resolution, 50 timesteps"
echo "----------------------------------------"
python run.py \
    --model thinkmorph_pat_ablation1 \
    --data AI2ThorPathTracing_10 \
    --work-dir ./outputs_ablation_study

# Ablation 2: Lower resolution (512x512, 25 timesteps)
echo ""
echo "[2/2] Ablation 2: 512x512 resolution, 25 timesteps"
echo "----------------------------------------"
python run.py \
    --model thinkmorph_pat_ablation2 \
    --data AI2ThorPathTracing_10 \
    --work-dir ./outputs_ablation_study

echo ""
echo "========================================"
echo "Ablation study completed!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Ablation 1 (1024x1024, 50 steps):"
echo "    Results: ./outputs_ablation_study/thinkmorph_pat_ablation1/"
echo "    Images:  /weka/.../viz_outputs/ablation1_1024_50steps/"
echo ""
echo "  - Ablation 2 (512x512, 25 steps):"
echo "    Results: ./outputs_ablation_study/thinkmorph_pat_ablation2/"
echo "    Images:  /weka/.../viz_outputs/ablation2_512_25steps/"
echo "========================================"

