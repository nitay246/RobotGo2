# Comments in English only
import cv2
import numpy as np

# Unitree SDK
from unitree_sdk2py.go2.video.video_client import VideoClient


class Camera:
    """
    Simple wrapper around Unitree VideoClient.
    - get_frame() returns a BGR OpenCV image (np.ndarray) or None on failure.
    - close() releases underlying resources.
    - Can be used as a context manager (with ... as cam:).
    """

    def __init__(self, timeout_sec: float = 2.0):
        self._client = VideoClient()
        # Set RPC timeout for image retrieval
        self._client.SetTimeout(timeout_sec)
        # Initialize transport
        self._client.Init()

    def get_frame(self):
        """Fetch a single JPEG frame and decode to BGR; return None if unavailable."""
        try:
            code, data = self._client.GetImageSample()
            if code != 0 or not data:
                return None

            # Decode JPEG buffer -> BGR image
            buf = np.frombuffer(bytes(data), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return None
            return img

        except Exception as e:
            print(f"[CAM] Error: {e}")
            return None

    def close(self):
        """Release camera resources (idempotent)."""
        try:
            self._client.Close()
        except Exception:
            pass

    # -------- Context manager helpers --------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
