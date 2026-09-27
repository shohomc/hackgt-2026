"""Live depth view for a Raspberry Pi 4. Press q to quit.

Brighter pixels are closer. This is not distance in meters.

The depth model shipped at 256x256, which is about 1 FPS on a Pi 4 when
only one CPU core is doing the work. This script:
  - gives the model all 4 cores, and stops OpenCV from using them too
  - runs the same model at 128x128, which is four times less image to process

    python3 depth_bench.py          # 128, faster, coarser
    python3 depth_bench.py 256      # sharper, about 1-2 FPS
"""

import os

# Set this before numpy, OpenCV, or ONNX start their own thread pools.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# OpenCV's own threads fight the depth model for the Pi's 4 cores.
cv2.setNumThreads(1)

MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
MODEL_PATH = Path("models/midas_v21_small_256.onnx")
CAMERA = 0
THREADS = 4


def download_model() -> None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the depth model (about 64 MB)...")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "hackgt-depth"})
    temporary = MODEL_PATH.with_suffix(".part")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(MODEL_PATH)


def model_at_size(size: int) -> Path:
    """Point the published 256x256 model at a smaller square image."""
    if size == 256:
        return MODEL_PATH
    destination = MODEL_PATH.with_name(f"midas_{size}.onnx")
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    import onnx
    from onnx import numpy_helper

    model = onnx.load(MODEL_PATH)
    for value in list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 4 and dims[1].dim_value == 3:
            dims[2].dim_value = size
            dims[3].dim_value = size

    def retarget(array):
        if array.ndim != 1 or array.size not in (2, 3, 4) or array.dtype.kind not in "iu":
            return None
        values = array.astype(int).tolist()
        if values[-2:] != [256, 256]:
            return None
        values[-2:] = [size, size]
        return np.array(values, dtype=array.dtype)

    for initializer in model.graph.initializer:
        updated = retarget(numpy_helper.to_array(initializer))
        if updated is not None:
            initializer.CopyFrom(numpy_helper.from_array(updated, initializer.name))

    onnx.save(model, destination)
    return destination


def start_session(size: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    for candidate in (size, 256) if size != 256 else (256,):
        try:
            path = model_at_size(candidate)
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            name = session.get_inputs()[0].name
            output = session.run(None, {name: np.zeros((1, 3, candidate, candidate), np.float32)})
            if np.squeeze(output[0]).shape[-1] != candidate:
                raise RuntimeError("model ignored the smaller size")
        except Exception:
            if candidate == 256:
                raise
            leftover = MODEL_PATH.with_name(f"midas_{candidate}.onnx")
            leftover.unlink(missing_ok=True)
            print("This model file cannot run at 128. Using 256 instead.")
            continue
        print(f"Depth model: {candidate}x{candidate}, {THREADS} CPU cores.")
        return session, name, candidate


def depth_color(session, input_name: str, frame, size: int):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
    depth = np.squeeze(session.run(None, {input_name: tensor})[0])
    low, high = float(depth.min()), float(depth.max())
    gray = ((depth - low) / (high - low + 1e-6) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.resize(colored, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    size = 128
    if len(sys.argv) > 1:
        if sys.argv[1] not in ("128", "256"):
            print("Use 128 (faster) or 256 (sharper). Example: python3 depth_bench.py 256")
            sys.exit(1)
        size = int(sys.argv[1])

    download_model()
    session, input_name, size = start_session(size)

    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Press q to quit.")
    fps = 0.0
    frames = 0
    window_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames.")
            break

        colored = depth_color(session, input_name, frame, size)
        view = np.hstack([frame, colored])
        frames += 1
        elapsed = time.perf_counter() - window_start
        if elapsed >= 1:
            fps = frames / elapsed
            print(f"{fps:.1f} FPS")
            frames = 0
            window_start = time.perf_counter()

        cv2.putText(view, f"{fps:.1f} FPS", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Depth", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
