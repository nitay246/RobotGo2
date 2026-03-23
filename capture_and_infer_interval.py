import os
import signal
import shutil
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from AppConfig import AppConfig
from camera import Camera
from system_init import SystemInit

# -------------------- Fixed configuration (no CLI params) --------------------
MODEL_PATH = "best.pt"
OUTPUT_FOLDER = "dataset_infer_annotated"
INTERVAL_SEC = 0.5
DURATION_SEC = 600.0
CONF = 0.25
IMGSZ = 640

# Motion tuning
MAX_VX = 0.36
MAX_WZ = 0.55
STEER_KP = 0.65
STEER_KD = 0.18
CENTER_DEADBAND = 0.10
CENTER_BIAS = 0.00
EX_FILTER_ALPHA = 0.25
PATH_ROI_TOP = 0.55
MIN_PATH_COVERAGE = 0.015
MOTION_SMOOTH_ALPHA = 0.20
CMD_TIMEOUT_SEC = 2.5


def warm_up_camera(cam: Camera, max_attempts: int = 10, sleep_sec: float = 0.1):
    frame = None
    for _ in range(max_attempts):
        frame = cam.get_frame()
        if frame is not None:
            return frame
        time.sleep(sleep_sec)
    return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def motion_executor(avoid_client, motion_state: dict, stop_evt: threading.Event, rate_hz: int = 25) -> None:
    dt = 1.0 / float(rate_hz)
    vx_curr = 0.0
    wz_curr = 0.0
    while not stop_evt.is_set():
        now = time.time()
        target_vx = float(motion_state.get("target_vx", 0.0))
        target_wz = float(motion_state.get("target_wz", 0.0))
        cmd_until = float(motion_state.get("cmd_until", 0.0))

        if now > cmd_until:
            target_vx = 0.0
            target_wz = 0.0

        # First-order smoothing to avoid abrupt start/stop jerks.
        vx_curr = (1.0 - MOTION_SMOOTH_ALPHA) * vx_curr + MOTION_SMOOTH_ALPHA * target_vx
        wz_curr = (1.0 - MOTION_SMOOTH_ALPHA) * wz_curr + MOTION_SMOOTH_ALPHA * target_wz

        try:
            avoid_client.Move(vx_curr, 0.0, wz_curr)
        except Exception as e:
            print(f"[MOTION] Error sending move command: {e}")
        time.sleep(dt)

    try:
        avoid_client.Move(0.0, 0.0, 0.0)
    except Exception:
        pass


def build_path_mask(result, frame_shape):
    if result is None or result.masks is None:
        return None

    frame_h, frame_w = frame_shape[:2]
    masks = result.masks.data.cpu().numpy()
    if masks.size == 0:
        return None

    classes = None
    if result.boxes is not None and hasattr(result.boxes, "cls"):
        classes = result.boxes.cls.cpu().numpy().astype(int)

    names = result.names if hasattr(result, "names") else {}
    path_class_ids = {
        cls_id
        for cls_id, cls_name in names.items()
        if str(cls_name).strip().lower() in {"path", "road", "floor", "walkway"}
    }

    combined = np.zeros((frame_h, frame_w), dtype=np.bool_)
    for i, mask in enumerate(masks):
        if classes is not None and len(classes) == len(masks) and path_class_ids:
            if int(classes[i]) not in path_class_ids:
                continue

        if mask.shape[0] != frame_h or mask.shape[1] != frame_w:
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        combined |= (mask > 0.5)

    if not combined.any() and len(masks) > 0 and not path_class_ids:
        # Fallback for single-class custom models with unnamed/unknown class labels.
        largest = max(masks, key=lambda m: float(np.sum(m > 0.5)))
        if largest.shape[0] != frame_h or largest.shape[1] != frame_w:
            largest = cv2.resize(largest, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        combined = largest > 0.5

    return combined if combined.any() else None


def compute_path_command(mask: np.ndarray, ctrl_state: dict, now_ts: float):
    h, w = mask.shape[:2]
    y0 = int(clamp(PATH_ROI_TOP, 0.0, 0.95) * h)
    roi = mask[y0:, :]
    if roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    coverage = float(np.mean(roi))
    if coverage < MIN_PATH_COVERAGE:
        return 0.0, 0.0, coverage, 0.0

    ys, xs = np.where(roi)
    if xs.size == 0:
        return 0.0, 0.0, coverage, 0.0

    cx = float(np.mean(xs))
    ex_raw = (cx - (w / 2.0)) / max(w / 2.0, 1.0)
    ex_raw -= CENTER_BIAS

    ex_prev_f = float(ctrl_state.get("ex_f", 0.0))
    ex_f = (1.0 - EX_FILTER_ALPHA) * ex_prev_f + EX_FILTER_ALPHA * ex_raw

    prev_ts = float(ctrl_state.get("prev_ts", now_ts))
    prev_ex = float(ctrl_state.get("prev_ex", ex_f))
    dt = max(now_ts - prev_ts, 1e-3)
    dex = (ex_f - prev_ex) / dt

    ctrl_state["ex_f"] = ex_f
    ctrl_state["prev_ex"] = ex_f
    ctrl_state["prev_ts"] = now_ts

    if abs(ex_f) < CENTER_DEADBAND:
        wz = 0.0
    else:
        wz = clamp(-(STEER_KP * ex_f + STEER_KD * dex), -MAX_WZ, MAX_WZ)

    speed_scale = clamp(coverage / 0.20, 0.22, 1.0)
    vx = MAX_VX * speed_scale
    turn_penalty = clamp(1.0 - 0.8 * abs(ex_f), 0.30, 1.0)
    vx *= turn_penalty

    return vx, wz, coverage, ex_f


def prepare_output_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INIT] Created output folder: {path}")
        return

    removed = 0
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            removed += 1
        else:
            os.remove(full_path)
            removed += 1
    print(f"[INIT] Emptied output folder: {path} (removed {removed} entries)")


def main() -> None:
    stop_event = threading.Event()
    motion_stop_event = threading.Event()
    motion_state = {"target_vx": 0.0, "target_wz": 0.0, "cmd_until": 0.0}
    ctrl_state = {"ex_f": 0.0, "prev_ex": 0.0, "prev_ts": time.time()}
    motion_thread = None

    def _handle_signal(signum, frame):
        print("\n[SYS] Ctrl+C detected - stopping and returning control to user...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Keep same robot-side init pattern used in your other scripts.
    print("[INIT] Initializing Unitree systems...")
    sys_init = SystemInit(AppConfig)
    state_manager, sport, avoid = sys_init.init_unitree()
    _ = (state_manager, sport, avoid)

    # Match yolo_follow behavior: ensure obstacle avoidance API path is enabled.
    try:
        avoid.UseRemoteCommandFromApi(True)
        avoid.SwitchSet(True)
        print("[INIT] Obstacle avoidance is ON.")
    except Exception as e:
        print(f"[WARN] Could not explicitly re-enable obstacle avoidance settings: {e}")

    motion_thread = threading.Thread(
        target=motion_executor,
        args=(avoid, motion_state, motion_stop_event),
        daemon=True,
    )
    motion_thread.start()
    print("[INIT] Path-follow motion mode enabled.")

    prepare_output_folder(OUTPUT_FOLDER)

    print(f"[INIT] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("[INIT] Connecting to Unitree camera...")
    cam = Camera(timeout_sec=AppConfig.CAM_TIMEOUT_SEC)

    first_frame = warm_up_camera(cam)
    if first_frame is None:
        cam.close()
        raise RuntimeError("Could not read any camera frame. Check robot connection.")

    # Validate model compatibility before countdown so failures are immediate.
    try:
        _ = model.predict(first_frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    except Exception as e:
        cam.close()
        print("[ERR] Preflight inference failed before timed capture started.")
        print("[ERR] This is usually a model-version mismatch (weights vs ultralytics/torch).")
        print(f"[ERR] Details: {e}")
        print("[HINT] Verify the same ultralytics version used to train/export this model.")
        return

    print("\n" + "=" * 56)
    print("  READY TO START TIMED CAPTURE + INFERENCE")
    print("  Starting now...")
    print("=" * 56)

    print("\n[RUN] Capture + inference started")
    print(f"[RUN] Interval: {INTERVAL_SEC:.2f}s")
    print(f"[RUN] Duration: {DURATION_SEC:.1f}s")
    print(f"[RUN] Expected captures: {int(DURATION_SEC / INTERVAL_SEC)}")
    print(f"[RUN] Continuous move: max_vx={MAX_VX:.2f} | max_wz={MAX_WZ:.2f} | timeout={CMD_TIMEOUT_SEC:.2f}s")
    print("[RUN] Press Ctrl+C to stop early\n")

    start_time = time.time()
    next_capture_time = start_time
    saved_count = 0

    try:
        while not stop_event.is_set():
            now = time.time()
            elapsed = now - start_time

            if elapsed >= DURATION_SEC:
                print("\n[RUN] Duration reached. Stopping automatically.")
                break

            if now >= next_capture_time:
                frame = cam.get_frame()
                if frame is None:
                    print(f"[{elapsed:05.1f}s] [WARN] Dropped frame.")
                    next_capture_time += INTERVAL_SEC
                    continue

                infer_t0 = time.time()
                result = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
                infer_ms = (time.time() - infer_t0) * 1000.0

                annotated = result.plot()
                det_count = 0 if result.boxes is None else len(result.boxes)
                path_mask = build_path_mask(result, frame.shape)

                vx_cmd, wz_cmd, coverage, ex = 0.0, 0.0, 0.0, 0.0
                if path_mask is not None:
                    vx_cmd, wz_cmd, coverage, ex = compute_path_command(path_mask, ctrl_state, now)

                motion_state["target_vx"] = vx_cmd
                motion_state["target_wz"] = wz_cmd
                motion_state["cmd_until"] = time.time() + CMD_TIMEOUT_SEC

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = f"infer_{timestamp}.jpg"
                out_path = os.path.join(OUTPUT_FOLDER, out_name)
                cv2.imwrite(out_path, annotated)

                saved_count += 1
                print(
                    f"[{elapsed:05.1f}s] Saved: {out_name} | det={det_count} | infer={infer_ms:.1f} ms"
                )
                print(
                    f"[{elapsed:05.1f}s] MoveCmd: vx={vx_cmd:.2f} wz={wz_cmd:.2f} | cov={coverage:.3f} ex={ex:.3f}"
                )

                next_capture_time += INTERVAL_SEC

            # Keep CPU usage low between capture events.
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[RUN] Interrupted by user.")
    finally:
        stop_event.set()
        motion_state["target_vx"] = 0.0
        motion_state["target_wz"] = 0.0
        motion_state["cmd_until"] = 0.0
        motion_stop_event.set()
        if motion_thread is not None:
            motion_thread.join(timeout=1.0)

        # Release command path back to user/remote side on exit.
        try:
            avoid.Move(0.0, 0.0, 0.0)
            avoid.UseRemoteCommandFromApi(False)
            print("[SYS] Motion stopped. API control released to user.")
        except Exception as e:
            print(f"[WARN] Could not fully release API control: {e}")

        cam.close()
        print(f"[DONE] Camera closed. Total annotated images saved: {saved_count}")
        print(f"[DONE] Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
