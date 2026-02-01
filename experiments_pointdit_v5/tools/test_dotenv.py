#!/usr/bin/env python
"""Test dotenv loading and wandb connection"""

import os
from pathlib import Path

# Simulate what train.py does
try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        print("✓ Loaded .env with dotenv")
except ImportError:
    print("✗ dotenv not available")

# Check if API key was loaded
api_key = os.getenv('WANDB_API_KEY')
if api_key:
    print(f"✓ WANDB_API_KEY loaded: {api_key[:10]}...")
else:
    print("✗ WANDB_API_KEY not found")

# Test wandb
try:
    import wandb
    print("✓ wandb imported")
    run = wandb.init(project="PointDiT", name="dotenv_test", reinit=True)
    print(f"✓ wandb.init() successful: {wandb.run.url}")
    wandb.log({'test': 123})
    wandb.finish()
    print("✓ All tests passed!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
