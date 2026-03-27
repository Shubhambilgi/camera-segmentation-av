import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Performance knobs for laptop real-time usage.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
INFER_EVERY_N_FRAMES = 3
USE_FP16_ON_CUDA = True


def resolve_paths():
    project_root = Path(__file__).resolve().parent
    sam2_repo_root = project_root / "sam2"
    if not sam2_repo_root.exists():
        raise RuntimeError("SAM2 repo not found at ./sam2")

    # Required to avoid SAM2 import-guard error.
    os.chdir(sam2_repo_root)
    sys.path.insert(0, str(sam2_repo_root))

    checkpoint_candidates = [
        project_root / "checkpoints" / "sam2_hiera_large.pt",
        sam2_repo_root / "checkpoints" / "sam2_hiera_large.pt",
    ]
    checkpoint = next((str(p) for p in checkpoint_candidates if p.exists()), None)
    if checkpoint is None:
        raise FileNotFoundError("Missing checkpoint: sam2_hiera_large.pt")

    config = "configs/sam2/sam2_hiera_l.yaml"
    return config, checkpoint


def blend_mask(frame_bgr, mask_bool, color=(0, 255, 0), alpha=0.45):
    overlay = frame_bgr.copy()
    overlay[mask_bool] = color
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)


def main():
    config, checkpoint = resolve_paths()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Starting SAM2 model load...")
    model = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    print("Model loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (camera index 0)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(
        "Controls: q=quit, p=pause, c=center prompt, "
        "left click=choose prompt point"
    )
    paused = False
    manual_point = None
    last_mask = None
    last_score = 0.0
    frame_idx = 0
    prev_tick = cv2.getTickCount()
    fps = 0.0

    window_name = "SAM2 Real-time Segmentation"
    cv2.namedWindow(window_name)

    def on_mouse(event, x, y, flags, param):
        nonlocal manual_point
        if event == cv2.EVENT_LBUTTONDOWN:
            manual_point = np.array([[x, y]], dtype=np.float32)

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            h, w = frame.shape[:2]
            point = (
                manual_point
                if manual_point is not None
                else np.array([[w // 2, h // 2]], dtype=np.float32)
            )
            labels = np.array([1], dtype=np.int32)

            run_infer = (frame_idx % INFER_EVERY_N_FRAMES == 0) or (last_mask is None)
            if run_infer:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.inference_mode():
                    if device == "cuda" and USE_FP16_ON_CUDA:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            predictor.set_image(frame_rgb)
                            masks, scores, _ = predictor.predict(
                                point_coords=point,
                                point_labels=labels,
                                multimask_output=False,
                            )
                    else:
                        predictor.set_image(frame_rgb)
                        masks, scores, _ = predictor.predict(
                            point_coords=point,
                            point_labels=labels,
                            multimask_output=False,
                        )
                last_mask = masks[0].astype(bool)
                last_score = float(scores[0])

            out = blend_mask(frame, last_mask, color=(0, 200, 0), alpha=0.45)

            px, py = int(point[0, 0]), int(point[0, 1])
            cv2.circle(out, (px, py), 6, (0, 0, 255), -1)

            current_tick = cv2.getTickCount()
            dt = (current_tick - prev_tick) / cv2.getTickFrequency()
            prev_tick = current_tick
            if dt > 0:
                fps = 1.0 / dt

            cv2.putText(
                out,
                (
                    f"SAM2 Fast | score={last_score:.3f} | {device} | "
                    f"fps={fps:.1f} | infer/frames={INFER_EVERY_N_FRAMES}"
                ),
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, out)
            frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("c"):
            manual_point = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
