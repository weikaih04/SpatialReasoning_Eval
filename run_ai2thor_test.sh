#!/bin/bash
# Test script for AI2Thor spatial reasoning evaluation
# Runs 4 evaluations with 10 samples each

set -e

# Create output directories
mkdir -p /weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/viz_outputs/thinkmorph_base
mkdir -p /weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/viz_outputs/thinkmorph_pat
mkdir -p /weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/viz_outputs/thinkmorph_pet

echo "========================================"
echo "Starting AI2Thor Spatial Evaluation"
echo "========================================"

# Test 1: ThinkMorph Base -> Path Tracing (10 samples)
echo ""
echo "[1/4] ThinkMorph Base -> AI2ThorPathTracing_10"
echo "----------------------------------------"
python run.py --model thinkmorph_base --data AI2ThorPathTracing_10 --work-dir ./outputs_ai2thor_test

# Test 2: ThinkMorph Base -> Perspective NoArrow (10 samples)
echo ""
echo "[2/4] ThinkMorph Base -> AI2ThorPerspective_NoArrow_10"
echo "----------------------------------------"
python run.py --model thinkmorph_base --data AI2ThorPerspective_NoArrow_10 --work-dir ./outputs_ai2thor_test

# Test 3: Path Tracing Finetuned -> Path Tracing (10 samples)
echo ""
echo "[3/4] ThinkMorph PAT -> AI2ThorPathTracing_10"
echo "----------------------------------------"
python run.py --model thinkmorph_pat --data AI2ThorPathTracing_10 --work-dir ./outputs_ai2thor_test

# Test 4: Perspective Taking Finetuned -> Perspective NoArrow (10 samples)
echo ""
echo "[4/4] ThinkMorph PET -> AI2ThorPerspective_NoArrow_10"
echo "----------------------------------------"
python run.py --model thinkmorph_pet --data AI2ThorPerspective_NoArrow_10 --work-dir ./outputs_ai2thor_test

echo ""
echo "========================================"
echo "All evaluations completed!"
echo "Results saved in: ./outputs_ai2thor_test"
echo "Visualization images saved in: /weka/.../viz_outputs/"
echo "========================================"

