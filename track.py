"""Track objects through a video with YOLO26n (NCNN) and ByteTrack.

NCNN finds the boxes. ByteTrack keeps the same ID on an object across frames.

First run downloads the weights and exports them. Later runs reuse that export.

    pip install ultralytics
    python3 track.py video.mp4

Press q to stop early.
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

NCNN_MODEL = Path("yolo26n_ncnn_model")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 track.py video.mp4")
        sys.exit(1)

    video = sys.argv[1]
    if not NCNN_MODEL.exists():
        print("Exporting YOLO26n to NCNN. This only happens once.")
        YOLO("yolo26n.pt").export(format="ncnn")

    model = YOLO(NCNN_MODEL)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Could not open {video}")
        sys.exit(1)

    seen = set()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
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

    cap.release()
    cv2.destroyAllWindows()
    print(f"Tracked {len(seen)} objects.")


if __name__ == "__main__":
    main()
