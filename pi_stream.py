"""Send each webcam frame to the laptop through the ngrok address it prints.

    pip install websockets
    python3 pi_stream.py wss://YOUR-TUNNEL.ngrok-free.app
    python3 pi_stream.py wss://YOUR-TUNNEL.ngrok-free.app 1
"""

import sys
import time

import cv2
from websockets.sync.client import connect

from camera_feed import open_camera


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 pi_stream.py wss://YOUR-TUNNEL.ngrok-free.app [camera]")
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
                with connect(
                    url,
                    max_size=8_000_000,
                    compression=None,
                    additional_headers={"ngrok-skip-browser-warning": "1"},
                ) as ws:
                    print("Connected. Streaming the camera at half size.")
                    sent = 0
                    last_report = time.perf_counter()
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            print("Camera stopped sending frames.")
                            return
                        # Half width and height before the bytes go through ngrok.
                        ok, jpg = cv2.imencode(
                            ".jpg",
                            cv2.pyrDown(frame),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                        )
                        if not ok:
                            continue
                        payload = jpg.tobytes()
                        ws.send(payload)
                        sent += len(payload)
                        if time.perf_counter() - last_report >= 5:
                            print(f"sent {sent / 1_048_576:.1f} MB this connection")
                            last_report = time.perf_counter()
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
