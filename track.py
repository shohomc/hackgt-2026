"""Track live webcam objects with YOLO26n (NCNN) and ByteTrack.

NCNN finds the boxes. ByteTrack keeps the same ID on an object across frames.

First run downloads the weights and exports them. Later runs reuse that export.
Run this on the Pi desktop or over VNC. Press q to quit.

    pip install ultralytics
    python3 track.py
    python3 track.py 1
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from camera_feed import open_camera

NCNN_MODEL = Path("yolo26n_ncnn_model")


def main() -> None:
    camera = 0
    if len(sys.argv) > 1:
        try:
            camera = int(sys.argv[1])
        except ValueError:
            print("Usage: python3 track.py [camera index]")
            sys.exit(1)

    if not NCNN_MODEL.exists():
        print("Exporting YOLO26n to NCNN. This only happens once.")
        YOLO("yolo26n.pt").export(format="ncnn")

    model = YOLO(NCNN_MODEL)
    cap = open_camera(camera)
    if not cap.isOpened():
        print(f"Could not open camera {camera}.")
        print("Plug in the webcam, then run: ls /dev/video*")
        sys.exit(1)

    # Keep one frame buffered so tracking follows the live view, not a backlog.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    seen = set()
    print("Tracking the webcam. Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

            if result.boxes.id is not None:
                seen.update(result.boxes.id.int().tolist())

            cv2.imshow("Track", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Tracked {len(seen)} objects.")


if __name__ == "__main__":
    main()
