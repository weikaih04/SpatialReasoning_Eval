#!/bin/bash
# Test script for MMCOT model evaluation (multi-choice QA with visual CoT)
# Runs evaluation with 10 samples for quick verification

set -e

# Configuration - modify checkpoint path here
MODEL_PATH="${MODEL_PATH:-/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/ckpt/mmcot/run_20260111_8gpu/0000380}"
RUN_NAME=$(basename $(dirname ${MODEL_PATH}))
STEP_NAME=$(basename ${MODEL_PATH})
SAVE_DIR="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/results/mmcot/${RUN_NAME}/${STEP_NAME}"

# Set environment variables for config.py
export THINKMORPH_MODEL_PATH="${MODEL_PATH}"
export THINKMORPH_SAVE_DIR="${SAVE_DIR}"

# Create output directory
mkdir -p ${SAVE_DIR}

echo "========================================"
echo "MMCOT Model Test Evaluation"
echo "========================================"
echo "Model Path: ${MODEL_PATH}"
echo "Save Dir: ${SAVE_DIR}"
echo "Dataset: AI2ThorPathTracing_10 (10 samples)"
echo "========================================"
echo ""

# Run evaluation on Path Tracing (mmcot is trained on path tracing data with multi-choice QA)
echo "[1/1] ThinkMorph MMCOT -> AI2ThorPathTracing_10"
echo "----------------------------------------"
python run.py --model thinkmorph_mmcot --data AI2ThorPathTracing_10 --work-dir ./outputs_mmcot_test --verbose

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results saved in: ./outputs_mmcot_test"
echo "Visualization images saved in: ${SAVE_DIR}"
echo "========================================"
