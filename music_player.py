# -------------------- Audio Class --------------------
import asyncio
import os
import threading
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc.contrib.media import MediaPlayer

class music_player:
    def __init__(self, ip="192.168.123.161"):
        self.ip = ip
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_background_loop, daemon=True)
        self.conn = None
        self.player = None
        self.is_playing = False
        self.thread.start()

    def _start_background_loop(self):
        """Runs the asyncio loop in a background thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())
        self.loop.run_forever()

    async def _connect(self):
        """Internal async connection setup."""
        try:
            print("[AUDIO] Connecting to WebRTC...")
            self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
            await self.conn.connect()
            print("[AUDIO] Connected.")
        except Exception as e:
            print(f"[AUDIO] Connection failed: {e}")

    async def _play_async(self, file_path):
        """Internal async play logic."""
        if self.is_playing or not self.conn:
            return
        
        try:
            self.player = MediaPlayer(file_path)
            if self.conn.pc:
                self.conn.pc.addTrack(self.player.audio)
                self.is_playing = True
                print(f"[AUDIO] Playing: {file_path}")
        except Exception as e:
            print(f"[AUDIO] Play error: {e}")

    async def _stop_async(self):
        """Internal async stop logic."""
        if self.player and self.is_playing:
            self.player = None # Releasing player stops stream
            self.is_playing = False
            print("[AUDIO] Stopped.")

    # --- Public Methods (Call these from your Main Loop) ---
    def play(self, filename="dora-doradura-mp3.mp3"):
        mp3_path = os.path.join(os.path.dirname(__file__), filename)
        asyncio.run_coroutine_threadsafe(self._play_async(mp3_path), self.loop)

    def stop(self):
        asyncio.run_coroutine_threadsafe(self._stop_async(), self.loop)