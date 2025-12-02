import signal
import time
import math
import threading
import cv2
import numpy as np
from ultralytics import YOLO

# Project imports
from AppConfig import AppConfig
from system_init import SystemInit
from target_lock import TargetLock
from camera import Camera

# -------------------- Globals --------------------
stop_event = threading.Event()

behavior = {
    "mode": "FOLLOW",  # FOLLOW, APPROACH, HOLD
    "vx": 0.0,
    "wz": 0.0,
    "until": 0.0,
    "cooldown_until": 0.0,
    "target_box": None,
    "roi_px": None,
}

# -------------------- Utilities --------------------
def handle_sigint(signum, frame):
    print("\n[SYS] Ctrl+C detected — stopping...")
    stop_event.set()

# -------------------- Main --------------------
def main():
    signal.signal(signal.SIGINT, handle_sigint)

    # ------------------------------------------------------------
    # Initialize all system components using SystemInit
    # ------------------------------------------------------------
    sys = SystemInit(AppConfig)

    state_manager, sport, avoid = sys.init_unitree(behavior)
    follower = sys.init_follower(state_manager, avoid, behavior, stop_event)
    follower.stop_event = stop_event  # attach the real stop_event

    cam, model, names = sys.init_vision()
    lock = sys.init_target_lock()

    print("[SYS] All systems initialized.")

    # -------------------- FPS --------------------
    fps_t0, frames = time.time(), 0

    # -------------------- HOLD logic --------------------
    hold_until = 0.0
    last_announce = 0.0

    # -------------------- YOLO + state machine loop --------------------
    try:
        while not stop_event.is_set():

            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            h, w = frame.shape[:2]

            # ROI in px
            rx1 = int(AppConfig.ROI_NORM[0] * w)
            ry1 = int(AppConfig.ROI_NORM[1] * h)
            rx2 = int(AppConfig.ROI_NORM[2] * w)
            ry2 = int(AppConfig.ROI_NORM[3] * h)
            roi_w = float(rx2 - rx1)
            roi_h = float(ry2 - ry1)
            roi_cx = 0.5 * (rx1 + rx2)

            # Draw ROI color
            roi_col = (0, 255, 0) if behavior["mode"] in ("APPROACH", "HOLD") else (255, 255, 255)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_col, 2)

            # YOLO inference
            res = model.predict(frame, imgsz=640, conf=AppConfig.MIN_CONF, verbose=False)[0]

            # Collect candidates (only backpacks)
            candidates = []
            if hasattr(res, 'boxes') and res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), cid, p in zip(boxes, clss, confs):
                    if names.get(int(cid), "") != "chair" or p < AppConfig.MIN_CONF:
                        continue
                    if (y2 - y1) < (AppConfig.MIN_BOX_FRAC * h):
                        continue
                    candidates.append((float(p), (float(x1), float(y1), float(x2), float(y2))))

            now = time.time()
            mode = behavior["mode"]

            # Share ROI with motion thread
            behavior["roi_px"] = (rx1, ry1, rx2, ry2)

            # -------------------- FOLLOW MODE --------------------
            if mode == "FOLLOW":
                if now - last_announce >= 0.5:
                    print("In FOLLOW")
                    last_announce = now
                if now >= behavior["cooldown_until"]:

                    if not lock.active and candidates:
                        got = lock.acquire(candidates, roi_rect=(rx1, ry1, rx2, ry2))
                        if got:
                            behavior["mode"] = "APPROACH"
                            behavior["target_box"] = lock.box
                            behavior["vx"] = 0.0
                            behavior["wz"] = 0.0

            # -------------------- APPROACH MODE --------------------
            if mode == "APPROACH":
                if now - last_announce >= 0.5:
                    print("In APROACH")
                    last_announce = now

                if lock.active:
                    lock.update(candidates)

                if not lock.active or lock.box is None:
                    behavior["mode"] = "FOLLOW"
                    behavior["target_box"] = None
                else:
                    x1, y1, x2, y2 = lock.box
                    cx = 0.5 * (x1 + x2)
                    bh = float(y2 - y1)
                    ex = (cx - roi_cx) / max(roi_w, 2)
                    print(ex)
                    size_ratio = bh / max(roi_h, 1.0)
                    ey = 1.0 - size_ratio

                    # Yaw
                    if abs(ex) < AppConfig.CENTER_TOL:
                        wz_t = 0.0
                    else:
                        wz_t = -ex

                    # Forward/back
                    if abs(ey) < AppConfig.SIZE_TOL:
                        behavior["vx"] = 0.0
                        behavior["wz"] = 0.0

                        hold_until = now + AppConfig.HOLD_SECONDS
                        behavior["mode"] = "HOLD"
                        print("[APPROACH] Target reached → HOLD")

                    elif ey > 0.0:
                        behavior["vx"] = AppConfig.K_VX_FWD * min(ey, 1.0)
                        behavior["wz"] = max(-AppConfig.MAX_WZ, min(AppConfig.MAX_WZ, wz_t))
                    else:
                        behavior["vx"] = -AppConfig.K_VX_BACK * min(-ey, 1.0)
                        behavior["wz"] = max(-AppConfig.MAX_WZ, min(AppConfig.MAX_WZ, wz_t))

                    behavior["target_box"] = lock.box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)

            # -------------------- HOLD MODE --------------------
            if mode == "HOLD":
                behavior["vx"] = 0.0
                behavior["wz"] = 0.0

                if now - last_announce >= 0.5:
                    print("Found — holding position…")
                    sport.Hello()
                    last_announce = now

                if now >= hold_until:
                    print("Found — returning to follow.")
                    behavior["mode"] = "FOLLOW"
                    behavior["cooldown_until"] = now + AppConfig.COOLDOWN_SECONDS

                    lock.reset()
                    behavior["target_box"] = None

            # Draw all YOLO detections
            if res is not None and res.boxes is not None:
                for b in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # FPS
            frames += 1
            if now - fps_t0 >= 1.0:
                fps = frames / (now - fps_t0)
                cv2.putText(frame, f"FPS: {fps:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255,255,255),
                            2,
                            cv2.LINE_AA)
                fps_t0 = now
                frames = 0

            cv2.imshow(AppConfig.WIN_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        avoid.Move(0.0, 0.0, 0.0)
        avoid.UseRemoteCommandFromApi(False)
        cam.close()
        cv2.destroyAllWindows()
        follower.join(timeout=1.0)
        print("[SYS] Shutdown complete.")


if __name__ == "__main__":
    main()
