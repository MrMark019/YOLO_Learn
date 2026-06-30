"""
室内盆栽植物检测 - 测试与推理脚本
"""

import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "runs/detect/indoor_potted_plant_pi/weights/best.pt"


def validate_model():
    """在测试集上验证模型"""
    print("=" * 60)
    print("模型验证（测试集）")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件不存在：{MODEL_PATH}")
        print("请先运行训练脚本 train_indoor_plant.py")
        return

    model = YOLO(MODEL_PATH)
    print(f"\n加载模型：{MODEL_PATH}")

    metrics = model.val(data="datasets/indoor_potted_plant/data.yaml", split="test")

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")
    print("=" * 60)

    return metrics


def predict_image(image_path):
    """单张图像推理"""
    print("=" * 60)
    print(f"图像推理：{image_path}")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件不存在：{MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    results = model(image_path, imgsz=416)
    result = results[0]

    print(f"\n检测到 {len(result.boxes)} 个盆栽:")
    for box in result.boxes:
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        print(f"  - potted plant: {confidence:.2%} "
              f"(x1:{bbox[0]:.1f}, y1:{bbox[1]:.1f}, x2:{bbox[2]:.1f}, y2:{bbox[3]:.1f})")

    output_path = "prediction_indoor_plant.jpg"
    result.save(filename=output_path)
    print(f"\n结果已保存：{output_path}")

    # 可选显示
    img = cv2.imread(output_path)
    if img is not None:
        cv2.imshow("Detection Result", img)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def predict_camera(camera_index: int = 0):
    """使用摄像头实时推理（树莓派可用），带 FPS、调试信息、自动保存"""
    print("=" * 60)
    print(f"摄像头实时推理（索引 {camera_index}）")
    print("=" * 60)
    print("按键说明：")
    print("  q - 退出")
    print("  s - 手动保存当前帧")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件不存在：{MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    # Windows 上使用 CAP_DSHOW 更稳定
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"错误：无法打开摄像头索引 {camera_index}")
        return

    # 创建保存目录
    save_dir = os.path.join("camera_captures", f"cam_{camera_index}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n检测到的帧将自动保存到: {save_dir}/")
    print(f"摄像头 {camera_index} 已启动，按 'q' 或 ESC 退出，'s' 手动保存当前帧...\n")

    window_name = "Indoor Plant Detection"
    frame_count = 0
    detect_count = 0
    fps = 0.0
    prev_time = time.time()
    last_log_time = prev_time

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

        # 确保内存连续
        frame = np.ascontiguousarray(frame)

        # 推理（降低 conf 阈值到 0.25，避免漏检）
        results = model(frame, imgsz=320, conf=0.25, verbose=False)
        result = results[0]
        num_boxes = len(result.boxes)

        # 绘制检测框和附加信息
        annotated = result.plot()
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f} | Plants: {num_boxes}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # 每 1 秒打印一次调试信息，减少刷屏
        if current_time - last_log_time >= 1.0:
            last_log_time = current_time
            confs = [float(box.conf[0]) for box in result.boxes]
            conf_str = ", ".join(f"{c:.2%}" for c in confs) if confs else "None"
            print(
                f"[Frame {frame_count:4d}] FPS={fps:.1f}  Plants={num_boxes}  Confidences=[{conf_str}]"
            )

        # 自动保存检测到的帧（每 30 帧最多保存一次，避免硬盘写爆）
        if num_boxes > 0:
            detect_count += 1
            if detect_count % 30 == 1:
                save_path = os.path.join(save_dir, f"auto_{frame_count:06d}.jpg")
                cv2.imwrite(save_path, annotated)
                print(f"  -> 自动保存: {save_path}")

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

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n摄像头已关闭。共处理 {frame_count} 帧，保存目录: {save_dir}/")


def export_model():
    """导出 ONNX / NCNN 格式（树莓派加速）"""
    print("=" * 60)
    print("模型导出")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件不存在：{MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)

    print("\n导出为 ONNX 格式...")
    model.export(format="onnx", imgsz=416)
    print(f"✓ ONNX 导出完成：{MODEL_PATH.replace('.pt', '.onnx')}")

    print("\n导出为 NCNN 格式（树莓派推荐）...")
    model.export(format="ncnn", imgsz=416)
    print(f"✓ NCNN 导出完成")


if __name__ == "__main__":
    print("\n室内盆栽植物检测 - 测试脚本")
    print("=" * 60)
    print("用法:")
    print("  python test_indoor_plant.py validate           # 测试集验证")
    print("  python test_indoor_plant.py predict <图>       # 单张推理")
    print("  python test_indoor_plant.py camera [index]     # 摄像头实时检测")
    print("  python test_indoor_plant.py export             # 导出 ONNX + NCNN")
    print("=" * 60)

    if len(sys.argv) < 2:
        validate_model()
    else:
        cmd = sys.argv[1]
        if cmd == "validate":
            validate_model()
        elif cmd == "predict" and len(sys.argv) > 2:
            predict_image(sys.argv[2])
        elif cmd == "camera":
            idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            predict_camera(idx)
        elif cmd == "export":
            export_model()
        else:
            print(f"\n未知命令：{cmd}")
            print("请使用：validate | predict | camera | export")
