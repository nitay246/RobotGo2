# Comments in English only
import time
import math
import threading
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class FollowConfig:
    # Motion smoothing and limits
    SMOOTH_ALPHA: float = 0.2
    MAX_VX: float = 0.40
    MAX_WZ: float = 0.96
    FOLLOW_DT: float = 0.04  # 25 Hz

    # UWB follow control
    DEAD_BAND_D: float = 1.2
    DIST_SLOWDOWN: float = 1.0
    DEAD_BAND_O: float = 0.20
    SLOWDOWN_ANGLE: float = math.radians(60)
    MAX_VX_FOLLOW: float = 0.9
    MAX_WZ_FOLLOW: float = 0.96


class FollowController:
    """
    Background follow controller that blends UWB-based velocities with a shared behavior state.
    External code sets behavior["mode"] and (optionally) behavior["vx"], behavior["wz"].
    """

    def __init__(
        self,
        state_manager: Any,
        avoid_client: Any,
        behavior: Dict[str, Any],
        config: FollowConfig | None = None,
    ):
        self.state_manager = state_manager
        self.avoid_client = avoid_client
        self.behavior = behavior
        self.cfg = config or FollowConfig()
        self._thread: threading.Thread | None = None

    def start(self, stop_event: threading.Event, daemon: bool = True) -> None:
        """Start the follow loop in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop, args=(stop_event,), daemon=daemon
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Join the background thread (optional)."""
        if self._thread:
            self._thread.join(timeout=timeout)

    # -------------------- Internal loop --------------------
    def _run_loop(self, stop_evt: threading.Event):
        vx_cmd, wz_cmd = 0.0, 0.0

        while not stop_evt.is_set():
            # --- Read UWB estimates ---
            dis = getattr(self.state_manager.remote_state, "distance_est", None)
            ori = getattr(self.state_manager.remote_state, "orientation_est", None)

            # Distance control (vx_follow)
            if dis is None:
                vx_follow = 0.0
            else:
                err_d = dis
                if abs(err_d) <= self.cfg.DEAD_BAND_D:
                    vx_follow = 0.0
                else:
                    scale = min(abs(err_d) / self.cfg.DIST_SLOWDOWN, 1.0)
                    vx_follow = math.copysign(self.cfg.MAX_VX_FOLLOW * scale, err_d)
                    vx_follow = max(-self.cfg.MAX_VX_FOLLOW, min(self.cfg.MAX_VX_FOLLOW, vx_follow))

            # Orientation control (wz_follow)
            if ori is None:
                wz_follow = 0.0
            else:
                err_o = ori
                if abs(err_o) <= self.cfg.DEAD_BAND_O:
                    wz_follow = 0.0
                else:
                    scale = min(abs(err_o) / self.cfg.SLOWDOWN_ANGLE, 1.0)
                    wz_follow = math.copysign(self.cfg.MAX_WZ_FOLLOW * scale, err_o)
                    wz_follow = max(-self.cfg.MAX_WZ_FOLLOW, min(self.cfg.MAX_WZ_FOLLOW, wz_follow))

            # --- Blend with behavior state ---
            mode = self.behavior.get("mode", "FOLLOW")
            if mode in ("APPROACH", "HOLD"):
                vx_t = float(self.behavior.get("vx", 0.0))
                wz_t = float(self.behavior.get("wz", 0.0))
            else:  # "FOLLOW"
                vx_t = vx_follow
                wz_t = wz_follow


            # Send command
            try:
                self.avoid_client.Move(vx_t, 0.0, wz_t)
            except Exception as e:
                print(f"[FOLLOW MOVE] Error: {e}")

            time.sleep(self.cfg.FOLLOW_DT)

        # Stop safely when loop exits
        self.avoid_client.Move(0.0, 0.0, 0.0)
