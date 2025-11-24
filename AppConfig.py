# app_config.py
import math

class AppConfig:
    # -------------------- Window / Camera --------------------
    WIN_NAME = "GO2 Camera + YOLO (Follow + Chair)"
    CAM_TIMEOUT_SEC = 2.0

    # -------------------- YOLO / ROI --------------------
    MIN_CONF = 0.35
    MIN_BOX_FRAC = 0.05
    ROI_NORM = (0.33, 0.1, 0.67, 0.8)
    SIZE_TOL = 0.08
    CENTER_TOL = 0.10

    # -------------------- Motion Control --------------------
    MAX_VX = 0.40     # m/s
    MAX_WZ = 0.96     # rad/s
    K_WZ = 1.2
    K_VX_FWD = 0.8
    K_VX_BACK = 0.4
    SMOOTH_ALPHA = 0.2
    FOLLOW_DT = 0.04

    # -------------------- Follow controller (UWB) --------------------
    DEAD_BAND_D = 1.2
    DIST_SLOWDOWN = 1.0
    DEAD_BAND_O = 0.20
    SLOWDOWN_ANGLE = math.radians(60)
    MAX_VX_FOLLOW = 0.9
    MAX_WZ_FOLLOW = 0.96

    # -------------------- Target lock --------------------
    LOCK_IOU_MIN = 0.25
    LOCK_MAX_MISS_FR = 10
    PREFER_ROI = True

    # -------------------- Behavior timing --------------------
    HOLD_SECONDS = 3.0
    COOLDOWN_SECONDS = 15.0

