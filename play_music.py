import logging
import asyncio
import os 
import sys
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc.contrib.media import MediaPlayer
import copy

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

async def main():
    try:
        # Choose a connection method (uncomment the correct one)
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.123.161")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
        
        f= conn.connect()

        
        mp3_path = os.path.join(os.path.dirname(__file__), "d.mp3")
        
        logging.info(f"Playing MP3: {mp3_path}")
        audio_track1 = MediaPlayer(mp3_path).audio   # Use MediaPlayer for MP3
        #audio_track2 = MediaPlayer(mp3_path).audio # Get the audio track from the player

        await f
        print("WebRTC connection established.")

        conn.pc.addTrack(audio_track1)
        audio_track1 = MediaPlayer(mp3_path).audio   # Use MediaPlayer for MP3
        ob= conn.pc.addTrack(audio_track1)

        ##conn.pc.addTrack(audio_track2)  # Add the audio track to the WebRTC connection

        await asyncio.sleep(3600)  # Keep the program running to handle events

    except ValueError as e:
        # Log any value errors that occur during the process.
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle Ctrl+C to exit gracefully.
        print("\nProgram interrupted by user")
        sys.exit(0)

