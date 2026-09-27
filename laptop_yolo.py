"""Receive live frames from the Pi and run YOLO26n here.

Start this before the Pi script. Both machines must be on the same network.

    pip install ultralytics websockets
    python3 laptop_yolo.py

Press q to quit.
"""

import queue
import socket
import threading

import cv2
import numpy as np
from ultralytics import YOLO
from websockets.sync.server import serve

PORT = 8765
frames: queue.Queue[bytes] = queue.Queue(maxsize=1)


def local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def handler(connection) -> None:
    print(f"Pi connected from {connection.remote_address[0]}")
    try:
        for message in connection:
            if frames.full():
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
            frames.put(message)
    except Exception as error:
        print(f"Pi disconnected ({error})")


def serve_frames() -> None:
    with serve(handler, "0.0.0.0", PORT, max_size=8_000_000, compression=None) as server:
        server.serve_forever()


def main() -> None:
    threading.Thread(target=serve_frames, daemon=True).start()
    print(f"Listening on {local_ip()}:{PORT}")
    print(f"On the Pi: python3 pi_stream.py ws://{local_ip()}:{PORT}")

    model = YOLO("yolo26n.pt")
    print("Waiting for frames. Press q to quit.")
    cv2.namedWindow("Track")

    while True:
        try:
            jpg = frames.get(timeout=0.03)
        except queue.Empty:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        cv2.imshow("Track", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
