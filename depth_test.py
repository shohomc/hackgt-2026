"""Show a depth map on the laptop GPU. Near colors are warmer. Press q to quit.

Values are meters. Pass a photo, or use the laptop webcam:

    python depth_test.py
    python depth_test.py photo.jpg
"""

import sys

import cv2
import torch
from ultralytics import YOLO


def show(model: YOLO, frame) -> None:
    result = model(frame, device=0, verbose=False)[0]
    cv2.imshow("Depth", result.plot())


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot see the GPU.")
    print(f"Depth model on {torch.cuda.get_device_name(0)}")
    model = YOLO("yolo26n-depth.pt")

    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            sys.exit(f"Could not read {sys.argv[1]}")
        show(model, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("No webcam. Pass a photo: python depth_test.py photo.jpg")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        show(model, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
