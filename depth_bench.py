"""Monocular depth benchmark for a Raspberry Pi 4B.

Uses Intel MiDaS v2.1 small (EfficientNet-Lite3, 256x256) through ONNX
Runtime. That model is a convolutional network sized for a Pi 4 CPU.
YOLO26n-depth is a dense head at a much larger input, and the Pi 4 has no
NPU, so quantizing it does not make it a real-time depth model.

The output is relative depth: larger values are closer. It is not distance
in meters.

Bench (writes runs/<timestamp>.json — these are the resume numbers):

    python3 depth_bench.py --bench --image photo.jpg

Live webcam preview (press q to quit, then the same JSON is written):

    python3 depth_bench.py

Pi setup:

    pip install onnxruntime
    # OpenCV with a window needs the apt build, not opencv-python-headless:
    # sudo apt install python3-opencv
"""

import argparse
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
INPUT_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MiDaS v2.1 small monocular depth.")
    parser.add_argument("--model", default="models/midas_v21_small_256.onnx")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", help="Still image instead of the webcam.")
    parser.add_argument("--bench", action="store_true", help="Time a fixed frame and exit.")
    parser.add_argument("--frames", type=int, default=20, help="Timed iterations after warmup.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4, help="Pi 4 has 4 CPU cores.")
    return parser.parse_args()


def download_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading MiDaS v2.1 small ({MODEL_URL})")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "hackgt-depth-bench"})
    tmp = path.with_suffix(".onnx.part")
    with urllib.request.urlopen(request) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(path)


def load_session(path: Path, threads: int):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    # Matches isl-org/MiDaS tf/run_onnx.py for model-small.onnx:
    # RGB in 0..1, stretched to 256x256, NCHW. Normalization is inside the graph.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    tensor = resized.astype(np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))[None]


def infer(session, input_name: str, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    tensor = preprocess(frame_bgr)
    preprocess_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    output = session.run(None, {input_name: tensor})[0]
    infer_ms = (time.perf_counter() - start) * 1000
    return np.squeeze(output).astype(np.float32), preprocess_ms, infer_ms


def colorize(depth: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    low, high = np.percentile(depth, [2, 98])
    scaled = np.clip((depth - low) / (high - low + 1e-6), 0, 1)
    colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return colored, (time.perf_counter() - start) * 1000


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * (q / 100)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": round(percentile(values, 50), 2),
        "p95_ms": round(percentile(values, 95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def memory_mb() -> dict[str, float | None]:
    rss = peak = None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = round(int(line.split()[1]) / 1024, 1)
            elif line.startswith("VmHWM:"):
                peak = round(int(line.split()[1]) / 1024, 1)
    return {"rss_mb": rss, "peak_rss_mb": peak}


def device_info(threads: int) -> dict:
    ram_mb = None
    model = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram_mb = round(int(line.split()[1]) / 1024)
                break
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("Model"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": model,
        "cpu_count": os.cpu_count(),
        "threads": threads,
        "ram_mb": ram_mb,
    }


def open_source(args: argparse.Namespace):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read {args.image}")
            sys.exit(1)
        return None, frame

    from camera_feed import open_camera

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        print("Plug in the webcam, or pass a still: python3 depth_bench.py --bench --image photo.jpg")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap, None


def grab(cap, still: np.ndarray | None) -> np.ndarray | None:
    if still is not None:
        return still
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def compose(frame: np.ndarray, depth_color: np.ndarray, infer_ms: float, depth_fps: float) -> np.ndarray:
    depth_view = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    view = np.hstack([frame, depth_view])
    scale = 960 / view.shape[1]
    if scale < 1:
        view = cv2.resize(view, (960, int(view.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    label = f"MiDaS small 256   infer {infer_ms:.0f} ms   depth {depth_fps:.2f} FPS   relative, not meters"
    cv2.rectangle(view, (0, 0), (view.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(view, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    infer = report["inference"]
    pipeline = report["pipeline"]
    print()
    print(f"model:     MiDaS v2.1 small, {report['model_size_mb']} MB, {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"device:    {report['device'].get('cpu_model') or report['device']['platform']}")
    print(f"threads:   {report['device']['threads']}    frames: {report['measured_frames']} after {report['warmup_frames']} warmup")
    print(f"inference: median {infer['median_ms']} ms   p95 {infer['p95_ms']} ms   ({report['model_only_fps']} FPS)")
    print(f"pipeline:  median {pipeline['median_ms']} ms   p95 {pipeline['p95_ms']} ms   ({report['pipeline_fps']} FPS)")
    print(f"memory:    rss {report['memory']['rss_mb']} MB   peak {report['memory']['peak_rss_mb']} MB")
    print("depth:     relative (higher = closer), not metric meters")
    print(f"saved:     {path}")


def build_report(args, model_path: Path, preprocess_ms, infer_ms, post_ms, pipeline_ms) -> dict:
    infer_stats = summarize(infer_ms)
    pipeline_stats = summarize(pipeline_ms)
    return {
        "task": "monocular_relative_depth",
        "model": "MiDaS v2.1 small",
        "checkpoint": model_path.name,
        "source": MODEL_URL,
        "input_hw": [INPUT_SIZE, INPUT_SIZE],
        "runtime": "onnxruntime CPUExecutionProvider",
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 1),
        "device": device_info(args.threads),
        "warmup_frames": args.warmup,
        "measured_frames": len(infer_ms),
        "preprocess": summarize(preprocess_ms),
        "inference": infer_stats,
        "postprocess": summarize(post_ms),
        "pipeline": pipeline_stats,
        "model_only_fps": round(1000 / infer_stats["median_ms"], 2) if infer_stats["median_ms"] else 0,
        "pipeline_fps": round(1000 / pipeline_stats["median_ms"], 2) if pipeline_stats["median_ms"] else 0,
        "memory": memory_mb(),
        "notes": [
            "Relative depth only. Larger values are closer. Do not describe this as meters.",
            "model_only_fps is session.run on one prepared tensor.",
            "pipeline_fps includes resize, inference, and colormap on that same frame.",
            "Quote the median and p95 from the Pi itself. Desktop numbers are not this result.",
        ],
    }


def bench(session, input_name: str, frame: np.ndarray, args, model_path: Path) -> None:
    preprocess_ms, infer_ms, post_ms, pipeline_ms = [], [], [], []
    total = args.warmup + args.frames
    for index in range(total):
        depth, pre, inf = infer(session, input_name, frame)
        colored, post = colorize(depth)
        if index < args.warmup:
            continue
        preprocess_ms.append(pre)
        infer_ms.append(inf)
        post_ms.append(post)
        pipeline_ms.append(pre + inf + post)
        print(f"  {index - args.warmup + 1}/{args.frames}  infer {inf:.0f} ms")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    preview = compose(frame, colored, infer_ms[-1], 1000 / infer_ms[-1])
    preview_path = Path("runs") / f"{stamp}_preview.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview)

    report = build_report(args, model_path, preprocess_ms, infer_ms, post_ms, pipeline_ms)
    report["preview"] = str(preview_path)
    write_report(Path("runs") / f"{stamp}.json", report)


def has_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def live(session, input_name: str, cap, still, args, model_path: Path) -> None:
    if not has_display():
        print("No desktop session. Timing a fixed frame instead. Use the desktop or VNC for a live window.")
        frame = grab(cap, still)
        if cap is not None:
            cap.release()
        if frame is None:
            print("Could not read a frame to benchmark.")
            sys.exit(1)
        bench(session, input_name, frame, args, model_path)
        return

    print("Depth preview running. Press q to quit and save the timing report.")
    preprocess_ms, infer_ms, post_ms, pipeline_ms = [], [], [], []
    warmed = 0
    last_view = None
    try:
        while True:
            started = time.perf_counter()
            frame = grab(cap, still)
            if frame is None:
                print("Stopped receiving frames.")
                break
            depth, pre, inf = infer(session, input_name, frame)
            colored, post = colorize(depth)
            pipeline = (time.perf_counter() - started) * 1000
            if warmed < args.warmup:
                warmed += 1
            else:
                preprocess_ms.append(pre)
                infer_ms.append(inf)
                post_ms.append(post)
                pipeline_ms.append(pipeline)
            fps = 1000 / pipeline if pipeline else 0
            last_view = compose(frame, colored, inf, fps)
            cv2.imshow("Depth", last_view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    if not infer_ms:
        print("Quit during warmup, so there is no timing report yet.")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if last_view is not None:
        preview_path = Path("runs") / f"{stamp}_preview.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), last_view)
    report = build_report(args, model_path, preprocess_ms, infer_ms, post_ms, pipeline_ms)
    report["notes"].append("Live run includes camera capture in pipeline_fps.")
    write_report(Path("runs") / f"{stamp}.json", report)


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.warmup < 0 or args.threads < 1:
        print("--frames must be >= 1, --warmup >= 0, and --threads >= 1.")
        sys.exit(1)

    model_path = Path(args.model)
    download_model(model_path)
    session, input_name = load_session(model_path, args.threads)

    cap, still = open_source(args)
    if args.bench:
        frame = grab(cap, still)
        if cap is not None:
            cap.release()
        if frame is None:
            print("Could not read a frame to benchmark.")
            sys.exit(1)
        print(f"Benchmarking {args.frames} frames on {args.threads} threads.")
        bench(session, input_name, frame, args, model_path)
        return

    live(session, input_name, cap, still, args, model_path)


if __name__ == "__main__":
    main()
