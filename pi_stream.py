"""Send each webcam frame to the laptop over a WebSocket.

No YOLO on the Pi. The laptop script receives these frames.

    pip install websockets
    python3 pi_stream.py ws://LAPTOP_IP:8765
    python3 pi_stream.py ws://LAPTOP_IP:8765 1
"""

import sys
import time

import cv2
from websockets.sync.client import connect

from camera_feed import open_camera


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 pi_stream.py ws://LAPTOP_IP:8765 [camera]")
        sys.exit(1)

    url = sys.argv[1]
    camera = 0
    if len(sys.argv) > 2:
        try:
            camera = int(sys.argv[2])
        except ValueError:
            print("Camera index must be a number.")
            sys.exit(1)

    cap = open_camera(camera)
    if not cap.isOpened():
        print(f"Could not open camera {camera}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while True:
            print(f"Connecting to {url}")
            try:
                with connect(url, max_size=8_000_000, compression=None) as ws:
                    print("Connected. Streaming the camera.")
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            print("Camera stopped sending frames.")
                            return
                        ok, jpg = cv2.imencode(
                            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                        )
                        if ok:
                            ws.send(jpg.tobytes())
            except KeyboardInterrupt:
                raise
            except Exception as error:
                print(f"Connection lost ({error}). Retrying...")
                time.sleep(1)
    finally:
        cap.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
