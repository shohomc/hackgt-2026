"""Receive live frames from the Pi and run YOLO26n here.

Start this before the Pi script. It opens a public ngrok address so the Pi
does not have to be on the same Wi-Fi. One-time login:

    ngrok config add-authtoken YOUR_TOKEN

The token is at https://dashboard.ngrok.com/get-started/your-authtoken

    pip install ultralytics websockets
    python3 laptop_yolo.py

Press q to quit.
"""

import json
import queue
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

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


def ngrok_exe() -> str | None:
    found = shutil.which("ngrok")
    if found:
        return found
    packages = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    matches = sorted(packages.glob("Ngrok.Ngrok_*/ngrok.exe")) if packages.exists() else []
    return str(matches[-1]) if matches else None


def read_tunnel() -> str | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1) as response:
            data = json.load(response)
    except Exception:
        return None
    for tunnel in data.get("tunnels", []):
        public = tunnel.get("public_url", "")
        if public.startswith("https://"):
            return "wss://" + public.removeprefix("https://")
    return None


def start_ngrok() -> subprocess.Popen | None:
    if read_tunnel():
        return None
    exe = ngrok_exe()
    if exe is None:
        print("ngrok is not installed. The Pi has to be on the same Wi-Fi.")
        return None
    process = subprocess.Popen(
        [exe, "http", str(PORT), "--log", "stdout"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        if read_tunnel():
            return process
        if process.poll() is not None:
            break
        time.sleep(0.25)
    process.terminate()
    print("ngrok did not open a tunnel. Sign in once, then run this script again:")
    print("  ngrok config add-authtoken YOUR_TOKEN")
    print("Get the token from https://dashboard.ngrok.com/get-started/your-authtoken")
    return None


def main() -> None:
    threading.Thread(target=serve_frames, daemon=True).start()
    ngrok = start_ngrok()
    public = read_tunnel()
    print(f"Listening locally on {local_ip()}:{PORT}")
    if public:
        print(f"On the Pi, from any network: python3 pi_stream.py {public}")
    else:
        print(f"On the Pi, same Wi-Fi only: python3 pi_stream.py ws://{local_ip()}:{PORT}")

    model = YOLO("yolo26n.pt")
    print("Waiting for frames. Press q to quit.")
    cv2.namedWindow("Track")

    try:
        run_viewer(model)
    finally:
        if ngrok is not None and ngrok.poll() is None:
            ngrok.terminate()
        cv2.destroyAllWindows()


def run_viewer(model: YOLO) -> None:
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


if __name__ == "__main__":
    main()
