import argparse
import signal
import time
from typing import Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from AppConfig import AppConfig
from camera import Camera
from system_init import SystemInit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a path-detection YOLO model (best.pt) on the Go2 camera feed."
    )
    parser.add_argument("--model", default="best.pt", help="Path to model weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--window", default="Go2 Path Segmentation Test", help="OpenCV window title."
    )
    return parser.parse_args()


def build_palette(names: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
    palette = {}
    for cls_id in names:
        # Stable pseudo-random class color for repeatable overlays.
        r = (37 * (cls_id + 3)) % 255
        g = (67 * (cls_id + 7)) % 255
        b = (97 * (cls_id + 11)) % 255
        palette[cls_id] = (int(b), int(g), int(r))
    return palette


def draw_segmentation_overlay(frame: np.ndarray, result, class_colors: Dict[int, Tuple[int, int, int]]):
    overlay = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    # Draw segmentation masks (pixel-level path detection).
    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(masks):
            cls_id = int(classes[i]) if i < len(classes) else 0
            color = class_colors.get(cls_id, (0, 255, 255))

            # Some models return masks in model-space, so align them to frame-space.
            if mask.shape[0] != frame_h or mask.shape[1] != frame_w:
                mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

            mask_bin = mask > 0.5
            overlay[mask_bin] = color

            # Optional contour to make edges easier to see.
            contours, _ = cv2.findContours(
                (mask_bin.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, color, 2)

        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # Draw boxes + class/conf text when available.
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        names = result.names if hasattr(result, "names") else {}

        for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confs):
            color = class_colors.get(int(cls_id), (0, 255, 255))
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
            label = f"{names.get(int(cls_id), str(cls_id))} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1i, max(20, y1i - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return frame


def main() -> None:
    args = parse_args()

    stop = {"value": False}

    def _handle_signal(signum, frame):
        stop["value"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("[INIT] Initializing Unitree systems...")
    sys_init = SystemInit(AppConfig)
    state_manager, sport, avoid = sys_init.init_unitree()
    _ = (state_manager, sport, avoid)  # keep references alive for runtime

    print(f"[INIT] Loading model: {args.model}")
    model = YOLO(args.model)
    names = model.model.names
    class_colors = build_palette(names)

    print("[INIT] Starting robot camera...")
    cam = Camera(timeout_sec=AppConfig.CAM_TIMEOUT_SEC)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, 1280, 720)

    print("[RUN] Live view started. Press 'q' to quit.")

    last_t = time.time()
    fps = 0.0

    try:
        while not stop["value"]:
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            vis = draw_segmentation_overlay(frame, result, class_colors)

            now = time.time()
            dt = max(now - last_t, 1e-6)
            last_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

            det_count = 0 if result.boxes is None else len(result.boxes)
            cv2.putText(
                vis,
                f"FPS: {fps:.1f} | Detections: {det_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(args.window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        print("[CLEANUP] Closing camera and windows.")
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
