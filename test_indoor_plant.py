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
    """使用摄像头实时推理，带窗口画面 + 控制台输出 + 自动保存"""
    # 树莓派/桌面 Linux 需要设置 DISPLAY 才能弹窗
    os.environ.setdefault("DISPLAY", ":0")

    print("=" * 60)
    print(f"摄像头实时推理（索引 {camera_index}）")
    print("=" * 60)
    print("窗口操作：")
    print("  窗口显示实时画面 + 检测框")
    print("  q 或 ESC = 退出")
    print("  s       = 手动保存当前帧")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件不存在：{MODEL_PATH}")
        return

    print(f"\n正在加载模型...")
    model = YOLO(MODEL_PATH)
    print("模型加载完成")

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
    print(f"截图保存目录: {save_dir}/")
    print(f"\n摄像头已启动！请对准盆栽观察窗口和控制台输出。\n")

    window_name = "Indoor Plant Detection (Press q to quit)"
    frame_count = 0
    detect_count = 0
    fps = 0.0
    avg_inference = 0.0
    prev_time = time.time()
    last_log_time = prev_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头画面")
                break

            frame_count += 1
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # 确保内存连续
            frame = np.ascontiguousarray(frame)

            # 推理
            t0 = time.time()
            results = model(frame, imgsz=320, conf=0.25, verbose=False)
            t1 = time.time()
            inf_time = t1 - t0
            if frame_count > 10:
                avg_inference = 0.95 * avg_inference + 0.05 * inf_time
            else:
                avg_inference = inf_time

            result = results[0]
            num_boxes = len(result.boxes)

            # 绘制检测框 + 画面信息
            annotated = result.plot()
            info_lines = [
                f"FPS: {fps:.1f}",
                f"Infer: {inf_time * 1000:.0f}ms",
                f"Plants: {num_boxes}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(
                    annotated, line, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            # 每秒打印一次控制台输出
            if current_time - last_log_time >= 1.0:
                last_log_time = current_time
                confs = [float(box.conf[0]) for box in result.boxes]
                conf_str = ("%.1f%%" % (max(confs) * 100)) if confs else "N/A"
                status = "DETECTED" if num_boxes > 0 else "scanning..."
                print(
                    f"[{frame_count:5d}] FPS={fps:5.1f} | "
                    f"Infer={inf_time*1000:4.0f}ms | "
                    f"Plants={num_boxes} | "
                    f"BestConf={conf_str:>6s} | {status}"
                )

            # 自动保存（检测到盆栽时每 30 个检测帧保存一次）
            if num_boxes > 0:
                detect_count += 1
                if detect_count % 30 == 1:
                    save_path = os.path.join(save_dir, f"auto_{frame_count:06d}.jpg")
                    cv2.imwrite(save_path, annotated)
                    print(f"  >> 自动保存: {save_path}")

            # 显示窗口
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n用户按 q/ESC 退出")
                break

            # 窗口被关闭（点击 X）
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n窗口被关闭")
                break

            # 手动保存
            if key == ord("s"):
                save_path = os.path.join(save_dir, f"manual_{frame_count:06d}.jpg")
                cv2.imwrite(save_path, annotated)
                print(f"  >> 手动保存: {save_path}")

    except KeyboardInterrupt:
        print("\nCtrl+C 退出")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n总计处理 {frame_count} 帧")
    print(f"平均推理时间: {avg_inference*1000:.0f}ms ({1/avg_inference:.1f} FPS)")
    print(f"截图保存目录: {save_dir}/")


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
