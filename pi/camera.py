"""Live USB webcam preview using OpenCV.

Run on the Pi desktop (or a VNC session). Press q to quit.
Pass a different camera index if /dev/video0 is not the webcam:

    python3 pi/camera.py 1
"""

import sys

import cv2


def open_camera(index: int) -> cv2.VideoCapture:
    # V4L2 is the Linux camera interface the Pi uses for USB webcams.
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


def main() -> None:
    index = 0
    if len(sys.argv) > 1:
        try:
            index = int(sys.argv[1])
        except ValueError:
            print("Camera index must be a number, for example: python3 pi/camera.py 0")
            sys.exit(1)

    cap = open_camera(index)
    if not cap.isOpened():
        print(f"Could not open camera {index}.")
        print("Plug in the webcam, then run: ls /dev/video*")
        sys.exit(1)

    print("Camera feed running. Press q in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stopped receiving frames from the camera.")
                break

            cv2.imshow("Camera", frame)  # Show a smaller image to fit on the screen.
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
