"""
NCNN 摄像头实时推理测试
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs/detect/indoor_potted_plant_pi/weights/best_ncnn_model"


def main(camera_index: int = 0):
    print("=" * 60)
    print(f"NCNN 摄像头实时推理（索引 {camera_index}）")
    print("=" * 60)
    print("按键：q=退出  s=手动保存")

    if not os.path.exists(MODEL_PATH):
        print(f"错误：NCNN 模型不存在：{MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH, task="detect")
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"错误：无法打开摄像头索引 {camera_index}")
        return

    save_dir = os.path.join("camera_captures", f"ncnn_cam_{camera_index}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n保存目录: {save_dir}/")
    print(f"摄像头 {camera_index} 已启动...\n")

    window_name = "Indoor Plant Detection (NCNN)"
    frame_count = 0
    fps = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头画面")
            break

        frame_count += 1
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

        # 确保内存连续，避免 NCNN 异常输出
        frame = np.ascontiguousarray(frame)

        results = model(frame, imgsz=320, conf=0.25, verbose=False)
        num_boxes = len(results[0].boxes)

        # 异常保护：NCNN 有时第一帧会出现大量乱框
        if num_boxes > 50:
            print(f"WARNING [Frame {frame_count}]: 异常检测数量 {num_boxes}，跳过绘制")
            annotated = frame.copy()
            cv2.putText(
                annotated,
                "NCNN ERROR: too many boxes",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        else:
            annotated = results[0].plot()

        cv2.putText(
            annotated,
            f"NCNN FPS: {fps:.1f} | Plants: {num_boxes}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

        # 检测窗口是否被关闭（点击 X 按钮）
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == ord("s"):
            save_path = os.path.join(save_dir, f"manual_{frame_count:06d}.jpg")
            cv2.imwrite(save_path, annotated)
            print(f"  -> 手动保存: {save_path}")

        # 自动保存有正常检测结果的帧
        if 0 < num_boxes <= 50 and frame_count % 30 == 1:
            save_path = os.path.join(save_dir, f"auto_{frame_count:06d}.jpg")
            cv2.imwrite(save_path, annotated)
            print(f"  -> 自动保存: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n已关闭。共处理 {frame_count} 帧")


if __name__ == "__main__":
    import sys

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)
