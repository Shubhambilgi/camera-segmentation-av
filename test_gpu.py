import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

print("=" * 50)
print("  SAM2 PROJECT - SYSTEM VERIFICATION")
print("=" * 50)

print(f"\nPython     : {sys.version[:6]}")
print(f"PyTorch    : {torch.__version__}")
print(f"OpenCV     : {cv2.__version__}")
print(f"NumPy      : {np.__version__}")

print(f"\nCUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM           : {vram:.1f} GB")
    print(f"CUDA Version   : {torch.version.cuda}")

project_root = Path(__file__).resolve().parent
sam2_repo_root = project_root / "sam2"
if not sam2_repo_root.exists():
    print("\nSAM2 repo       : MISSING (expected ./sam2)")
    sys.exit(1)

# Run from inside repo root so SAM2 package guard does not fail.
os.chdir(sam2_repo_root)
sys.path.insert(0, str(sam2_repo_root))

try:
    from sam2.build_sam import build_sam2

    print("\nSAM2           : importable OK")
except Exception as e:
    print(f"\nSAM2           : FAILED ({e})")

# Check checkpoint locations (project root or sam2 repo checkpoints).
candidate_ckpts = [
    project_root / "checkpoints" / "sam2_hiera_large.pt",
    sam2_repo_root / "checkpoints" / "sam2_hiera_large.pt",
]
found = [p for p in candidate_ckpts if p.exists()]
if found:
    size = found[0].stat().st_size / 1e6
    print(f"Checkpoint     : found {found[0]} ({size:.0f} MB)")
else:
    print("Checkpoint     : MISSING (sam2_hiera_large.pt)")

print("\n" + "=" * 50)
print("  ALL CHECKS DONE")
print("=" * 50)
