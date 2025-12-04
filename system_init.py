# system_init.py

import time
import cv2
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbState_
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient

from uwb_state_manager import UwbStateManager
from uwb_button_monitor import UwbButtonMonitor

from follow_controller import FollowConfig, FollowController
from camera import Camera
from target_lock import TargetLockConfig, TargetLock


class SystemInit:
    """
    Centralized initializer for all subsystems:
      - Unitree communications
      - UWB state manager
      - Sport + Obstacle Avoid clients
      - FollowController thread
      - Camera + YOLO model
      - Target locking
    """

    def __init__(self, config):
        """
        config: your AppConfig object (constants only)
        """
        self.cfg = config

    # ------------------------------------------------------------
    # UNITREE (UWB + SPORT + AVOID + BUTTON MONITOR)
    # ------------------------------------------------------------
    def init_unitree(self, behavior):
        print("[INIT] Initializing Unitree SDK...")

        ChannelFactoryInitialize(0)

        # UWB / state manager
        state_manager = UwbStateManager()

        button_monitor = UwbButtonMonitor(
            state_manager, 
            lambda: print("[UWB] Shutdown button pressed.")
        )

        uwb_sub = ChannelSubscriber("rt/uwbstate", UwbState_)
        uwb_sub.Init(button_monitor.get_callback(), 10)

        # Sport + obstacle clients
        sport = SportClient()
        avoid = ObstaclesAvoidClient()
        avoid.Init()
        sport.Init()
        avoid.UseRemoteCommandFromApi(True)
        avoid.SwitchSet(True)

        print("[INIT] Unitree communication established.")

        # Return handles
        return state_manager, sport, avoid

    # ------------------------------------------------------------
    # FOLLOW CONTROLLER THREAD
    # ------------------------------------------------------------
    def init_follower(self, state_manager, avoid, behavior, stop_event):
        print("[INIT] Starting FollowController thread...")

        follow_cfg = FollowConfig(
            SMOOTH_ALPHA=self.cfg.SMOOTH_ALPHA,
            MAX_VX=self.cfg.MAX_VX,
            MAX_WZ=self.cfg.MAX_WZ,
            FOLLOW_DT=self.cfg.FOLLOW_DT,
            DEAD_BAND_D=self.cfg.DEAD_BAND_D,
            DIST_SLOWDOWN=self.cfg.DIST_SLOWDOWN,
            DEAD_BAND_O=self.cfg.DEAD_BAND_O,
            SLOWDOWN_ANGLE=self.cfg.SLOWDOWN_ANGLE,
            MAX_VX_FOLLOW=self.cfg.MAX_VX_FOLLOW,
            MAX_WZ_FOLLOW=self.cfg.MAX_WZ_FOLLOW,
        )

        follower = FollowController(state_manager, avoid, behavior, follow_cfg)
        follower.start(stop_event, daemon=True)

        print("[INIT] FollowController is running.")
        return follower

    # ------------------------------------------------------------
    # CAMERA + YOLO
    # ------------------------------------------------------------
    def init_vision(self):
        print("[INIT] Initializing camera...")

        cam = Camera(timeout_sec=self.cfg.CAM_TIMEOUT_SEC)

        print("[INIT] Loading YOLO model...")
        model = YOLO("yolov8n.pt")    # original literal
        names = model.model.names

        print("[INIT] Creating display window...")
        # cv2.namedWindow(self.cfg.WIN_NAME, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(self.cfg.WIN_NAME, 960, 540)

        return cam, model, names

    # ------------------------------------------------------------
    # TARGET LOCK
    # ------------------------------------------------------------
    def init_target_lock(self):
        print("[INIT] Setting up TargetLock...")

        lock_cfg = TargetLockConfig(
            lock_iou_min=self.cfg.LOCK_IOU_MIN,
            lock_max_miss_fr=self.cfg.LOCK_MAX_MISS_FR,
            prefer_roi=self.cfg.PREFER_ROI,
        )
        lock = TargetLock(lock_cfg)
        return lock
