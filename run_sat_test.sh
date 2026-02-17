#!/bin/bash
# Test script for SAT spatial reasoning evaluation
# Runs evaluation on SAT_circular dataset with 10 samples

set -e

# Activate conda environment
source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/etc/profile.d/conda.sh
conda activate thinkmorph_eval

# Create output directories
mkdir -p /weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/viz_outputs/thinkmorph_pet_4940_sat

echo "========================================"
echo "Starting SAT Spatial Evaluation"
echo "========================================"

# Test: ThinkMorph PET (checkpoint 4940) -> SAT_circular (10 samples)
echo ""
echo "[1/1] ThinkMorph PET (ckpt 4940) -> SAT_circular_10"
echo "----------------------------------------"
python run.py \
  --data SAT_circular_10 \
  --model thinkmorph_pet \
  --work-dir ./outputs_sat_test \
  --verbose

echo ""
echo "========================================"
echo "SAT evaluation completed!"
echo "Results saved in: ./outputs_sat_test"
echo "Visualization images saved in: /weka/.../viz_outputs/thinkmorph_pet_4940_sat/"
echo "========================================"

