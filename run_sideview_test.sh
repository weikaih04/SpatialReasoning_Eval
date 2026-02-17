#!/bin/bash
# Test script for Sideview model evaluation (visual CoT description only)
# Runs evaluation with 10 samples for quick verification

set -e

# Configuration - modify checkpoint path here
MODEL_PATH="${MODEL_PATH:-/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/ckpt/sideview/run_20260111_8gpu/0000380}"
RUN_NAME=$(basename $(dirname ${MODEL_PATH}))
STEP_NAME=$(basename ${MODEL_PATH})
SAVE_DIR="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/results/sideview/${RUN_NAME}/${STEP_NAME}"

# Set environment variables for config.py
export THINKMORPH_MODEL_PATH="${MODEL_PATH}"
export THINKMORPH_SAVE_DIR="${SAVE_DIR}"

# Create output directory
mkdir -p ${SAVE_DIR}

echo "========================================"
echo "Sideview Model Test Evaluation"
echo "========================================"
echo "Model Path: ${MODEL_PATH}"
echo "Save Dir: ${SAVE_DIR}"
echo "Dataset: AI2ThorPathTracing_10 (10 samples)"
echo "========================================"
echo ""

# Run evaluation on Path Tracing (sideview is trained on path tracing data)
echo "[1/1] ThinkMorph Sideview -> AI2ThorPathTracing_10"
echo "----------------------------------------"
python run.py --model thinkmorph_sideview --data AI2ThorPathTracing_10 --work-dir ./outputs_sideview_test --verbose

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results saved in: ./outputs_sideview_test"
echo "Visualization images saved in: ${SAVE_DIR}"
echo "========================================"
