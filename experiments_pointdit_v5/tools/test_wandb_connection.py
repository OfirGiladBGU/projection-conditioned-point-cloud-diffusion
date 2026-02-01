#!/usr/bin/env python
"""Test wandb connection with API key from .env"""

import os
import re
import sys

# Parse .env manually
with open('.env', 'r') as f:
    content = f.read()
    match = re.search(r'WANDB_API_KEY="([^"]+)"', content)
    if match:
        api_key = match.group(1)
        os.environ['WANDB_API_KEY'] = api_key
        print(f"✓ Loaded API key: {api_key[:10]}...")
    else:
        print("✗ Could not find WANDB_API_KEY in .env")
        sys.exit(1)

# Test wandb connection
try:
    import wandb
    print("✓ wandb module imported")
    
    # Initialize and test
    run = wandb.init(project="PointDiT", name="test_connection", reinit=True)
    print(f"✓ wandb initialized: {wandb.run.name}")
    print(f"  Project: {wandb.run.project}")
    print(f"  URL: {wandb.run.url}")
    
    # Log dummy data
    wandb.log({'test_metric': 42, 'test_loss': 0.123})
    print("✓ Test metrics logged")
    
    wandb.finish()
    print("✓ Connection successful! Ready for training.")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
