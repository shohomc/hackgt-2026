"""Show a depth view of the Pi camera. Frames arrive through ngrok. Press q to quit.

Colors use the model's normal depth coloring.
ngrok_usage.txt stores how many frame bytes have been received.

    python laptop_yolo.py
"""

import json
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth
from websockets.sync.server import serve

PORT = 8765
USAGE = Path("ngrok_usage.txt")
latest: queue.Queue[bytes] = queue.Queue(maxsize=1)
usage_lock = threading.Lock()
session_bytes = 0
total_bytes = 0
last_saved = 0


def load_usage() -> int:
    try:
        return int(USAGE.read_text().strip())
    except (OSError, ValueError):
        return 0


def remember(nbytes: int) -> None:
    global session_bytes, total_bytes, last_saved
    with usage_lock:
        session_bytes += nbytes
        total_bytes += nbytes
        if total_bytes - last_saved >= 1_000_000:
            USAGE.write_text(str(total_bytes))
            last_saved = total_bytes


def receive(connection) -> None:
    print("Pi connected.")
    try:
        for message in connection:
            remember(len(message))
            if latest.full():
                latest.get_nowait()
            latest.put(message)
    except Exception as error:
        print(f"Pi disconnected ({error})")


def listen() -> None:
    with serve(receive, "127.0.0.1", PORT, max_size=8_000_000, compression=None) as server:
        server.serve_forever()


def tunnel_url() -> str | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1) as response:
            payload = json.load(response)
    except Exception:
        return None
    for tunnel in payload.get("tunnels", []):
        url = tunnel.get("public_url", "")
        if url.startswith("https://"):
            return "wss://" + url.removeprefix("https://")
    return None


def open_tunnel() -> subprocess.Popen | None:
    if tunnel_url():
        return None
    ngrok = shutil.which("ngrok")
    if ngrok is None:
        sys.exit("ngrok is not on PATH.")
    process = subprocess.Popen(
        [ngrok, "http", str(PORT), "--log", "stdout"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    for _ in range(40):
        if tunnel_url():
            return process
        if process.poll() is not None:
            break
        time.sleep(0.25)
    process.terminate()
    sys.exit("ngrok did not open a tunnel. Check: ngrok config add-authtoken YOUR_TOKEN")


def label(image: np.ndarray, depth: np.ndarray) -> np.ndarray:
    valid = depth[depth > 0]
    near = float(valid.min()) if valid.size else 0.0
    far = float(valid.max()) if valid.size else 0.0
    with usage_lock:
        session_mb = session_bytes / 1_048_576
        total_mb = total_bytes / 1_048_576
    text = f"{near:.1f}-{far:.1f} m    {session_mb:.1f} MB this run    {total_mb:.1f} MB through ngrok"
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    global total_bytes, last_saved
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot see the GPU.")
    torch.zeros(1, device="cuda")
    total_bytes = last_saved = load_usage()
    print(f"Depth model on {torch.cuda.get_device_name(0)}")
    print(f"Already received {total_bytes / 1_048_576:.1f} MB through ngrok")

    threading.Thread(target=listen, daemon=True).start()
    ngrok = open_tunnel()
    print(f"On the Pi: python3 pi_stream.py {tunnel_url()}")
    model = YOLO("yolo26n-depth.pt")
    print("Waiting for frames. Press q to quit.")

    try:
        while True:
            try:
                jpg = latest.get(timeout=0.03)
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            small = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if small is None:
                continue
            depth = np.squeeze(model(small, device=0, verbose=False)[0].depth.data.cpu().numpy())
            depth = cv2.resize(depth, (small.shape[1], small.shape[0]))
            colored = colorize_depth(depth)
            cv2.imshow("Depth", label(cv2.pyrUp(colored), depth))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        with usage_lock:
            USAGE.write_text(str(total_bytes))
        if ngrok is not None and ngrok.poll() is None:
            ngrok.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
