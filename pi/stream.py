"""Send each webcam frame to the laptop through the ngrok address it prints.

Directions that come back are written to the Arduino on /dev/ttyACM0.

    pip install websockets pyserial
    python3 pi/stream.py wss://YOUR-TUNNEL.ngrok-free.app
    python3 pi/stream.py wss://YOUR-TUNNEL.ngrok-free.app 1
"""

import sys
import time

import cv2
from websockets.sync.client import connect

from camera import open_camera

CONNECT_TIMEOUT = 3 * 60


def open_arduino():
    """USB serial to the chariot sketch. None when the board is absent."""
    try:
        import serial
    except ImportError:
        print("pip install pyserial to drive the Arduino.", flush=True)
        return None
    for path in ("/dev/ttyACM0", "/dev/ttyUSB0"):
        try:
            link = serial.Serial(path, 9600, timeout=0)
        except Exception:
            continue
        time.sleep(2)
        print(f"Arduino on {path}", flush=True)
        return link
    print("No Arduino on /dev/ttyACM0 or /dev/ttyUSB0.", flush=True)
    return None


def forward_commands(ws, link) -> None:
    while True:
        try:
            message = ws.recv(timeout=0)
        except TimeoutError:
            return
        if not isinstance(message, str):
            continue
        command = message.strip()
        if not command:
            continue
        print(f"Arduino: {command}", flush=True)
        if link is None:
            continue
        try:
            link.write((command + "\n").encode())
        except Exception as error:
            print(f"Arduino write failed ({error})", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 pi/stream.py wss://YOUR-TUNNEL.ngrok-free.app [camera]")
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
    link = open_arduino()

    disconnected_since = time.monotonic()
    try:
        while True:
            waited = time.monotonic() - disconnected_since
            if waited >= CONNECT_TIMEOUT:
                print("No connection within 3 minutes. Stopping.")
                return
            print(f"Connecting to {url}")
            try:
                with connect(
                    url,
                    max_size=8_000_000,
                    compression=None,
                    open_timeout=CONNECT_TIMEOUT - waited,
                    additional_headers={"ngrok-skip-browser-warning": "1"},
                ) as ws:
                    print("Connected. Streaming the camera at half size.")
                    disconnected_since = None
                    while True:
                        forward_commands(ws, link)
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
                        if ok:
                            ws.send(jpg.tobytes())
            except KeyboardInterrupt:
                raise
            except Exception as error:
                if disconnected_since is None:
                    disconnected_since = time.monotonic()
                if time.monotonic() - disconnected_since >= CONNECT_TIMEOUT:
                    print("No connection within 3 minutes. Stopping.")
                    return
                print(f"Connection lost ({error}). Retrying...")
                time.sleep(1)
    finally:
        cap.release()
        if link is not None:
            link.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
