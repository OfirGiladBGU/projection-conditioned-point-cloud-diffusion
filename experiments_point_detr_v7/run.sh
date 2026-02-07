#!/bin/bash
# Run Point-RT training script

cd /groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments_point_detr_v7

# Test script first
echo "Running quick tests..."
python test.py

if [ $? -eq 0 ]; then
    echo "Tests passed! Starting training..."
    python train.py
else
    echo "Tests failed!"
    exit 1
fi
