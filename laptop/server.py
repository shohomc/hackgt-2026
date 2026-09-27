"""Show the ten most probable FastSAM masks, colored by depth. Press q to quit.

    python laptop/server.py
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import torch
from typesafe_sdk import Choice, Noul, TypeSafeClient
from ultralytics import FastSAM, YOLO
from websockets.sync.server import serve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8765
TOP_MASKS = 10
NMS_IOU = 0.5
CONF = 0.25
USE_LOCAL = False
latest: queue.Queue[bytes] = queue.Queue(maxsize=1)


def load_env() -> None:
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def receive(connection) -> None:
    print("Pi connected.")
    try:
        for message in connection:
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


def across(value: float, width: int) -> float:
    """Horizontal position from 0 at the left edge to 1 at the right edge."""
    return round(min(max(value / width, 0.0), 1.0), 2)


def side_name(x: float, width: int) -> str:
    if x < width / 3:
        return "left"
    if x > width * 2 / 3:
        return "right"
    return "center"


TURN = Noul(
    instructions=(
        "Should the person turn at all? "
        "Yes only when a collision looks immediate: a mask persists across the frames, sits in the path ahead, "
        "and its average_depth_m is near that frame's depth_p10_m. "
        "No when the center is clear or the mask is nearer depth_p90_m."
    ),
    criteria={
        "true": "A turn should be made. A persisted obstacle is in the path and a collision looks immediate.",
        "false": "No turn. Keep the current heading. Nothing immediate is in the path.",
    },
)

STOP = Noul(
    instructions=(
        "Is a stop preferred over continuing? "
        "Yes when the path ahead changes suddenly: a close mask appears that was absent in the older frames, "
        "or a mask already in the path jumps to the close end. "
        "No when the person can keep walking, including by turning."
    ),
    criteria={
        "true": "Stop. The path ahead changed suddenly.",
        "false": "Keep navigating. Continuing, straight or by a turn, is preferred.",
    },
)

DIRECTION = Choice(
    instructions=(
        "Which way should the person turn? "
        "The image is the person's view. An obstacle on the left side of the image means turn right. "
        "An obstacle on the right side of the image means turn left. "
        "x_center is 0 at the left edge and 1 at the right edge. "
        "A soft turn is about 20 degrees. A hard turn is a sharp turn of about 45 degrees. "
        "Use a soft turn when the obstacle stays on one side and leaves the center open. "
        "Use a hard turn when it reaches into the center, around x 0.5."
    ),
    criteria={
        "hard left": "Turn sharply left, about 45 degrees. The obstacle is on the right and reaches into the center of the image.",
        "soft left": "Turn left about 20 degrees. The obstacle is on the right and leaves the center open.",
        "soft right": "Turn right about 20 degrees. The obstacle is on the left and leaves the center open.",
        "hard right": "Turn sharply right, about 45 degrees. The obstacle is on the left and reaches into the center of the image.",
    },
)

pending: queue.Queue[tuple[dict, int]] = queue.Queue(maxsize=1)
advice = "waiting"
advice_lock = threading.Lock()
request_id = 0


recent_frames: deque[dict] = deque(maxlen=8)


def valid_depth(depth: np.ndarray) -> np.ndarray:
    return depth[np.isfinite(depth) & (depth > 0)]


def average_depth(depth: np.ndarray) -> float | None:
    valid = valid_depth(depth)
    if valid.size == 0:
        return None
    return round(float(valid.mean()), 2)


def depth_percentiles(depth: np.ndarray) -> tuple[float | None, float | None]:
    """Close end (10th percentile) and far end (90th percentile) of one depth map."""
    valid = valid_depth(depth)
    if valid.size == 0:
        return None, None
    low, high = np.percentile(valid, [10, 90])
    return round(float(low), 2), round(float(high), 2)


def depth_map(model: YOLO, frame: np.ndarray) -> np.ndarray:
    depth = np.squeeze(model(frame, device=0, verbose=False)[0].depth.data.cpu().numpy())
    if depth.shape[:2] != frame.shape[:2]:
        depth = cv2.resize(
            depth.astype(np.float32),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return depth


def mask_depth(depth: np.ndarray, mask: np.ndarray) -> float | None:
    """Average depth of the pixels inside one mask."""
    if mask.shape[:2] != depth.shape[:2]:
        mask = cv2.resize(mask.astype(np.float32), (depth.shape[1], depth.shape[0]))
    return average_depth(depth[mask > 0.5])


def frame_objects(frame: np.ndarray, result, depth: np.ndarray) -> list[dict]:
    """Describe the strongest FastSAM masks in one camera frame."""
    height, width = frame.shape[:2]
    objects = []
    boxes = result.boxes
    if boxes is not None and len(boxes):
        mask_data = None if result.masks is None else result.masks.data.cpu().numpy()
        for index, xyxy in enumerate(boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            if mask_data is not None and index < len(mask_data):
                mean_depth = mask_depth(depth, np.squeeze(mask_data[index]))
            else:
                mean_depth = average_depth(
                    depth[
                        max(int(y1), 0) : min(int(y2), height),
                        max(int(x1), 0) : min(int(x2), width),
                    ]
                )
            center_x = (x1 + x2) / 2
            objects.append(
                {
                    "score": round(float(boxes.conf[index]), 2),
                    "side": side_name(center_x, width),
                    "x_center": across(center_x, width),
                    "x_min": across(x1, width),
                    "x_max": across(x2, width),
                    "area_fraction": round((x2 - x1) * (y2 - y1) / (width * height), 3),
                    "average_depth_m": mean_depth,
                }
            )
    objects.sort(key=lambda item: (item["average_depth_m"] is None, item["average_depth_m"] or 0))
    for rank, item in enumerate(objects, start=1):
        item["distance_rank"] = rank
    close_m, far_m = depth_percentiles(depth)
    third = max(width // 3, 1)
    recent_frames.append(
        {
            "depth_p10_m": close_m,
            "depth_p90_m": far_m,
            "depth_by_side_m": {
                "left": average_depth(depth[:, :third]),
                "center": average_depth(depth[:, third : third * 2]),
                "right": average_depth(depth[:, third * 2 :]),
            },
            "objects": objects,
            "direction_given": given_direction(),
        }
    )
    return objects


def frame_vote(frame: dict) -> str | None:
    """One frame's direction, or None when this frame is not obvious."""
    close_m = frame["depth_p10_m"]
    far_m = frame["depth_p90_m"]
    if close_m is None or far_m is None:
        return None
    span = far_m - close_m
    if span < 0.15:
        return None
    cutoff = close_m + 0.35 * span
    blocked = {"left": False, "center": False, "right": False}
    for obj in frame["objects"]:
        depth = obj["average_depth_m"]
        if depth is None or depth > cutoff or obj["area_fraction"] < 0.08:
            continue
        blocked[obj["side"]] = True
        if obj["side"] == "center" and obj["area_fraction"] >= 0.4:
            return "stop"
    if not blocked["center"]:
        return "straight"
    if blocked["left"] and not blocked["right"]:
        return "right"
    if blocked["right"] and not blocked["left"]:
        return "left"
    return None


def local_direction() -> str | None:
    """Direction when the last three frames agree. None means ask Jev."""
    frames = list(recent_frames)
    if len(frames) < 3:
        return None
    votes = [frame_vote(frame) for frame in frames[-3:]]
    if any(vote is None for vote in votes) or len(set(votes)) != 1:
        return None
    return votes[0]


def jev_state() -> dict:
    """The last several frames, with an explicit warning that detection can glitch."""
    return {
        "detection_warning": (
            f"These are the {TOP_MASKS} highest-scoring FastSAM masks in each frame, after dropping masks that overlap a higher-scoring one. They have no class name. "
            "objects are ordered closest first. distance_rank 1 is the closest mask and higher ranks are farther away. "
            "A mask can appear for one frame and vanish on the next. "
            "depth_p10_m is the closer tenth of the image and depth_p90_m is the farther tenth. "
            "Each mask has average_depth_m over its pixels. "
            "x_center is 0 at the left edge of the image and 1 at the right edge. x_min and x_max are the mask's left and right edges on that scale. "
            "An obstacle on the left side of the image means turn right to avoid it. An obstacle on the right side of the image means turn left to avoid it. "
            "A soft left or soft right is a turn of about 20 degrees. A hard left or hard right is a sharp turn of about 45 degrees. "
            "Use a soft turn when the obstacle stays on one side and leaves the center open. Use a hard turn when it reaches into the center, around x 0.5. "
            "Prefer a soft turn, a hard turn, or straight so the person keeps navigating. "
            "A hard or soft turn fits when a collision looks immediate: the mask persists, it sits in the path ahead, "
            "and its average_depth_m is near that frame's depth_p10_m. "
            "A mask near depth_p90_m is farther away. "
            "Stop fits a sudden change: a close mask appears that was absent in the older frames, "
            "or a mask already in the path jumps to the close end. "
            "A smaller value means closer, but the scale can jump, so compare depths within these frames. "
            "Frames are listed oldest first and newest last. "
            "direction_given is the direction already shown to the person for that frame. "
            "Keep it unless the scene has changed."
        ),
        "frames_oldest_first": list(recent_frames),
    }


def jev_current(stamp: int) -> bool:
    with advice_lock:
        return stamp == request_id


def jev_worker() -> None:
    global advice
    client = TypeSafeClient()
    while True:
        state, stamp = pending.get()
        try:
            turn = client.system_one(state=state, questions={"turn": TURN}).nouls["turn"]
            if not jev_current(stamp):
                continue
            stop = client.system_one(state=state, questions={"stop": STOP}).nouls["stop"]
            if not jev_current(stamp):
                continue
            turn_yes = turn.noul >= 0.5
            stop_yes = stop.noul >= 0.5
            print(
                f"turn {'yes' if turn_yes else 'no'} {turn.noul:.0%}  stop {'yes' if stop_yes else 'no'} {stop.noul:.0%}",
                flush=True,
            )
            if stop_yes:
                text = f"stop  {stop.noul:.0%}"
            elif not turn_yes:
                text = f"straight  {1 - turn.noul:.0%}"
            else:
                answer = client.system_one(state=state, questions={"direction": DIRECTION}).choices["direction"]
                if not jev_current(stamp):
                    continue
                text = f"{answer.choice}  {answer.confidence:.0%}"
        except Exception as error:
            text = "Jev unavailable"
            print(f"Jev: {error}", flush=True)
        print(text, flush=True)
        with advice_lock:
            if stamp == request_id:
                advice = text


def current_advice() -> str:
    with advice_lock:
        return advice


def given_direction() -> str | None:
    """The direction on screen, without its confidence."""
    text = current_advice()
    for name in ("hard left", "hard right", "soft left", "soft right", "straight", "stop"):
        if text.startswith(name):
            return name
    return None


def depth_color(depth_m: float | None, close_m: float | None, far_m: float | None) -> tuple[int, int, int]:
    """BGR tint from red (close) through yellow to blue (far) within one frame."""
    if depth_m is None or close_m is None or far_m is None or far_m <= close_m:
        return (0, 255, 255)
    span = far_m - close_m
    t = min(max((depth_m - close_m) / span, 0.0), 1.0)
    if t < 0.5:
        mix = t / 0.5
        return (0, int(255 * mix), 255)
    mix = (t - 0.5) / 0.5
    return (int(255 * mix), int(255 * (1 - mix)), int(255 * (1 - mix)))


def overlaps(masks: torch.Tensor | None, boxes: torch.Tensor, index: int, kept: list[int], iou_thresh: float) -> bool:
    """True when this region mostly covers one already kept."""
    if masks is not None:
        candidate = masks[index]
        for previous in kept:
            union = torch.logical_or(candidate, masks[previous]).sum()
            if union > 0 and float(torch.logical_and(candidate, masks[previous]).sum() / union) >= iou_thresh:
                return True
        return False
    candidate = boxes[index]
    for previous in kept:
        other = boxes[previous]
        x1 = torch.maximum(candidate[0], other[0])
        y1 = torch.maximum(candidate[1], other[1])
        x2 = torch.minimum(candidate[2], other[2])
        y2 = torch.minimum(candidate[3], other[3])
        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_a = torch.clamp(candidate[2] - candidate[0], min=0) * torch.clamp(candidate[3] - candidate[1], min=0)
        area_b = torch.clamp(other[2] - other[0], min=0) * torch.clamp(other[3] - other[1], min=0)
        union = area_a + area_b - inter
        if union > 0 and float(inter / union) >= iou_thresh:
            return True
    return False


def keep_top(result, limit: int = TOP_MASKS, iou_thresh: float = NMS_IOU):
    """Keep the highest-scoring masks, dropping ones that overlap a stronger mask."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return result
    masks = None if result.masks is None else result.masks.data > 0.5
    kept: list[int] = []
    for index in torch.argsort(boxes.conf, descending=True).tolist():
        if overlaps(masks, boxes.xyxy, index, kept, iou_thresh):
            continue
        kept.append(index)
        if len(kept) == limit:
            break
    order = torch.tensor(kept, device=boxes.conf.device)
    result.boxes = boxes[order]
    if result.masks is not None:
        result.masks = result.masks[order]
    return result


def paint(frame: np.ndarray, result, depth: np.ndarray) -> np.ndarray:
    """Tint each mask by its average depth. Red is close, blue is far. Farther masks are drawn first."""
    view = frame.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return view
    close_m, far_m = depth_percentiles(depth)
    height, width = frame.shape[:2]
    mask_data = None if result.masks is None else result.masks.data.cpu().numpy()
    layers = []
    for index, xyxy in enumerate(boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        if mask_data is not None and index < len(mask_data):
            mean_depth = mask_depth(depth, np.squeeze(mask_data[index]))
        else:
            mean_depth = average_depth(
                depth[
                    max(int(y1), 0) : min(int(y2), height),
                    max(int(x1), 0) : min(int(x2), width),
                ]
            )
        layers.append((mean_depth if mean_depth is not None else far_m or 0, index, xyxy, mean_depth))
    for _, index, xyxy, mean_depth in sorted(layers, key=lambda item: item[0], reverse=True):
        x1, y1, x2, y2 = (int(value) for value in xyxy)
        color = depth_color(mean_depth, close_m, far_m)
        if mask_data is not None and index < len(mask_data):
            mask = np.squeeze(mask_data[index])
            if mask.shape[:2] != view.shape[:2]:
                mask = cv2.resize(mask.astype(np.float32), (view.shape[1], view.shape[0]))
            region = mask > 0.5
            tint = np.array(color, dtype=np.float32)
            view[region] = (view[region].astype(np.float32) * 0.45 + tint * 0.55).astype(np.uint8)
        else:
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        label = "?.??m" if mean_depth is None else f"{mean_depth:.2f}m"
        cv2.putText(view, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return view


def main() -> None:
    global advice, request_id
    load_env()
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot see the GPU.")
    torch.zeros(1, device="cuda")
    print(f"FastSAM-x on {torch.cuda.get_device_name(0)}", flush=True)

    threading.Thread(target=listen, daemon=True).start()
    ngrok = open_tunnel()
    print(f"On the Pi: python3 pi/stream.py {tunnel_url()}", flush=True)
    segmenter = FastSAM(os.path.join(ROOT, "models", "FastSAM-x.pt"))
    depth_model = YOLO(os.path.join(ROOT, "models", "yolo26l-depth.pt"))
    if os.environ.get("TYPESAFE_API_KEY"):
        threading.Thread(target=jev_worker, daemon=True).start()
    else:
        advice = "set TYPESAFE_API_KEY"
        print("Set TYPESAFE_API_KEY to ask Jev for a direction.", flush=True)
    print(f"Showing the {TOP_MASKS} most probable masks, colored by depth. Press q to quit.", flush=True)
    asked_at = 0.0

    try:
        while True:
            try:
                jpg = latest.get(timeout=0.03)
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            segments = keep_top(segmenter(frame, device=0, imgsz=640, conf=CONF, verbose=False)[0])
            depth = depth_map(depth_model, frame)
            objects = frame_objects(frame, segments, depth)
            names = ", ".join(f"{item['score']:.2f} {item['average_depth_m']}m" for item in objects)
            latest_frame = recent_frames[-1]
            choice = local_direction() if USE_LOCAL else None
            now = time.time()
            route = ""
            if choice is not None:
                with advice_lock:
                    request_id += 1
                    advice = choice
                route = f"  local {choice}"
            elif os.environ.get("TYPESAFE_API_KEY") and now - asked_at >= 1:
                asked_at = now
                with advice_lock:
                    request_id += 1
                    stamp = request_id
                if pending.full():
                    try:
                        pending.get_nowait()
                    except queue.Empty:
                        pass
                pending.put((jev_state(), stamp))
            print(
                f"p10 {latest_frame['depth_p10_m']}m  p90 {latest_frame['depth_p90_m']}m  {names or 'nothing'}{route}",
                flush=True,
            )
            shown = cv2.pyrUp(paint(frame, segments, depth))
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 48), (0, 0, 0), -1)
            cv2.putText(
                shown,
                current_advice(),
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.imshow("FastSAM", shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if ngrok is not None and ngrok.poll() is None:
            ngrok.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
