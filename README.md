# camera-segmentation-av

Camera-based segmentation using Meta SAM2.

## Project layout expected

- `sam2/` -> cloned [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- `checkpoints/sam2_hiera_large.pt` -> model checkpoint
- `sam2_env/` -> Python virtual environment

## Verify setup

From `sam2_project`:

```powershell
& ".\sam2_env\Scripts\python.exe" ".\test_gpu.py"
& ".\sam2_env\Scripts\python.exe" ".\test_sam2.py"
```

## Real-time webcam segmentation

```powershell
& ".\sam2_env\Scripts\python.exe" ".\realtime_camera_sam2.py"
```

Controls:
- `q` quit
- `p` pause/resume
- mouse left-click to set prompt point
- `c` return to center-point prompt
