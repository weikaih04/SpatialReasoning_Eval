#!/bin/bash
# Script to run maze visualization test with 2 GPUs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thinkmorph_eval

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1

# Checkpoint paths
CKPT1="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/models/ThinkMorph-7B"
CKPT2="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/models/BAGEL-7B-MoT"

# Run test
cd /weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training

python test_maze_viz.py \
    --ckpt1 "$CKPT1" \
    --ckpt2 "$CKPT2" \
    --num-samples 10 \
    --dataset VSP_maze_task_main_original \
    --work-dir ./maze_test_results

echo "Test completed! Check results in ./maze_test_results"

