import signal
import time
import threading
import cv2
from concurrent.futures import ThreadPoolExecutor
import logging
import asyncio

# Project imports
from AppConfig import AppConfig
from system_init import SystemInit

logger = logging.getLogger(__name__)

# -------------------- Globals --------------------
audio_hub = None
audio_loop = None # New global to hold the background event loop

def start_audio_service():
    """
    Runs in a separate thread. Initializes the connection and keeps 
    the event loop running forever so the connection doesn't drop.
    """
    global audio_hub, audio_loop
    
    # Imports specific to this scope
    from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
    from go2_webrtc_driver.webrtc_audiohub import WebRTCAudioHub
    
    # Create a new event loop for this thread
    audio_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(audio_loop)
    
    # specific IP for Go2
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.123.161")

    try:
        # Connect and initialize
        audio_loop.run_until_complete(conn.connect())
        audio_hub = WebRTCAudioHub(conn, logger)
        print("[AUDIO] Service started and connected.")
        
        # Keep this loop running indefinitely to handle audio tasks
        audio_loop.run_forever()
    except Exception as e:
        print(f"[AUDIO] Error in audio thread: {e}")
    finally:
        # cleanup if loop stops
        if conn:
            audio_loop.run_until_complete(conn.disconnect())

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
    
    # Gracefully stop the audio loop
    global audio_loop
    if audio_loop and audio_loop.is_running():
        # Schedule the loop to stop from a thread-safe context
        audio_loop.call_soon_threadsafe(audio_loop.stop)

def bark():
    """
    Thread-safe bark function. 
    Sends the play command to the background audio thread.
    """
    global audio_hub, audio_loop
    
    if audio_hub is None or audio_loop is None:
        print("[AUDIO] Audio system not ready yet.")
        return

    # The coroutine we want to run
    async def play_coro():
        # UUID for the specific sound
        task = audio_hub.play_by_uuid('01315020-c95f-45b3-a29e-388d2bffbb2d')
        # Wait for it with a timeout inside the async loop
        await asyncio.wait_for(task, timeout=2.0)
        return "Done"

    try:    
        # Submit the work to the background thread
        future = asyncio.run_coroutine_threadsafe(play_coro(), audio_loop)
        
        # Wait for the result on the main thread (blocking for max 3 seconds)
        # This 3.0 allows for the 2.0s audio timeout + 1.0s buffer
        result = future.result(timeout=3.0) 
        # print(f"[AUDIO] Result: {result}") # Optional: debug print
        
    except (asyncio.TimeoutError, TimeoutError):
        # This catches if the main thread waited too long for the background thread
        print(f"[AUDIO] Timeout occurred after 2 seconds (Bark Skipped)")
    except Exception as e:
        print(f"[AUDIO] Error triggering bark: {e}")

# -------------------- Main --------------------
def main():
    # --- MODIFIED: Start Audio in Background Thread ---
    audio_thread = threading.Thread(target=start_audio_service, daemon=True)
    audio_thread.start()

    # Wait briefly for audio to initialize (optional, but prevents 'None' errors immediately)
    print("[SYS] Waiting for audio connection...")
    while audio_hub is None:
        time.sleep(0.1)
    # --------------------------------------------------

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
                
                # --- MODIFIED: Call new bark function ---
                # We removed the arguments because it uses globals internally now
                bark() 
                # ----------------------------------------

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
     
        # cv2.imshow(AppConfig.WIN_NAME, frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break 


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