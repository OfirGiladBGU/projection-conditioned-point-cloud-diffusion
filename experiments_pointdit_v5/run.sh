#!/bin/bash
# Quick start script for Experiment 4: Point-DiT

set -e

echo "============================================"
echo "Experiment 4: Point-DiT for Neural Stippling"
echo "============================================"
echo ""

# Activate environment
echo "Activating pc2 environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pc2

cd /groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments4

# Test model
echo ""
echo "Testing Point-DiT architecture..."
python model/point_dit.py

# Test dataset
echo ""
echo "Testing dataset (first sample)..."
python -c "
from dataset import FastStipplingDataset
dataset = FastStipplingDataset(
    source_dir='/groups/asharf_group/ofirgila/ControlNet/training/fill50k-gs/source',
    num_points=2048,
)
print(f'✓ Dataset ready: {len(dataset)} images')
sample = dataset[0]
print(f'✓ Image shape: {sample[\"image\"].shape}')
print(f'✓ Points shape: {sample[\"points\"].shape}')
"

# Start training
echo ""
echo "============================================"
echo "Starting training..."
echo "============================================"
echo ""
echo "Config:"
echo "  - Model: Point-DiT (17M parameters)"
echo "  - Dataset: FastStipplingDataset (50k images)"
echo "  - Points: 2048 per image"
echo "  - Batch size: 8"
echo "  - Epochs: 100"
echo ""
echo "Outputs will be saved to: ./outputs_exp4/"
echo ""
read -p "Press Enter to start training (or Ctrl+C to cancel)..."

python train.py
