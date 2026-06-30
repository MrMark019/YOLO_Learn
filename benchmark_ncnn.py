"""
PyTorch vs NCNN 速度对比测试
"""

import time
import numpy as np
import cv2
from ultralytics import YOLO


def benchmark(model_path, name, imgsz=320, runs=100):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Model: {model_path}")
    print(f"Image size: {imgsz}x{imgsz}, Runs: {runs}")
    print("=" * 60)

    model = YOLO(model_path, task="detect")

    # 生成一个随机测试帧
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        model(dummy_frame, imgsz=imgsz, verbose=False)

    # Benchmark
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(dummy_frame, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    avg_ms = times.mean() * 1000
    fps = 1.0 / times.mean()
    print(f"  Average: {avg_ms:.1f} ms/frame")
    print(f"  FPS:     {fps:.1f}")
    print(f"  Min/Max: {times.min()*1000:.1f} / {times.max()*1000:.1f} ms")

    return avg_ms, fps


if __name__ == "__main__":
    pt_ms, pt_fps = benchmark(
        "runs/detect/indoor_potted_plant_pi/weights/best.pt",
        "PyTorch (best.pt)",
        imgsz=320,
        runs=100,
    )

    ncnn_ms, ncnn_fps = benchmark(
        "runs/detect/indoor_potted_plant_pi/weights/best_ncnn_model",
        "NCNN (best_ncnn_model)",
        imgsz=320,
        runs=100,
    )

    print(f"\n{'='*60}")
    print("Summary")
    print("=" * 60)
    print(f"  PyTorch: {pt_ms:.1f} ms  ({pt_fps:.1f} FPS)")
    print(f"  NCNN:    {ncnn_ms:.1f} ms  ({ncnn_fps:.1f} FPS)")
    if ncnn_fps > pt_fps:
        print(f"  Speedup: {ncnn_fps/pt_fps:.2f}x faster with NCNN")
    else:
        print(f"  Note: NCNN is optimized for ARM; on x64 PC it may not be faster")
    print("=" * 60)
