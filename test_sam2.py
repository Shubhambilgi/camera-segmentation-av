import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

project_root = Path(__file__).resolve().parent
sam2_repo_root = project_root / "sam2"
if not sam2_repo_root.exists():
    raise RuntimeError("SAM2 repo not found at ./sam2")

# SAM2 raises an error if run from the parent of the cloned repo.
os.chdir(sam2_repo_root)
sys.path.insert(0, str(sam2_repo_root))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint_candidates = [
    project_root / "checkpoints" / "sam2_hiera_large.pt",
    sam2_repo_root / "checkpoints" / "sam2_hiera_large.pt",
]
checkpoint_path = next((str(p) for p in checkpoint_candidates if p.exists()), None)
if checkpoint_path is None:
    raise FileNotFoundError("Missing checkpoint: sam2_hiera_large.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam2("configs/sam2/sam2_hiera_l.yaml", checkpoint_path, device=device)
predictor = SAM2ImagePredictor(model)

# Create a synthetic image so this script does not depend on external files like test.jpg.
image = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.rectangle(image, (420, 220), (880, 600), (255, 255, 255), -1)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

input_point = np.array([[640, 360]])
input_label = np.array([1])
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print(f"SUCCESS: SAM2 running on {device}")
print(f"Predicted masks: {masks.shape}, best score: {float(scores.max()):.4f}")