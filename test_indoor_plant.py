"""
室内盆栽植物检测 - 测试与推理脚本
"""

import os
import sys
import cv2
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


def predict_camera():
    """使用摄像头实时推理（树莓派可用）"""
    print("=" * 60)
    print("摄像头实时推理（按 q 退出）")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件不存在：{MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n摄像头已启动，按 'q' 退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # imgsz=320 在树莓派上更快
        results = model(frame, imgsz=320, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Indoor Plant Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已关闭")


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
    print("  python test_indoor_plant.py validate     # 测试集验证")
    print("  python test_indoor_plant.py predict <图> # 单张推理")
    print("  python test_indoor_plant.py camera       # 摄像头实时检测")
    print("  python test_indoor_plant.py export       # 导出 ONNX + NCNN")
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
            predict_camera()
        elif cmd == "export":
            export_model()
        else:
            print(f"\n未知命令：{cmd}")
            print("请使用：validate | predict | camera | export")
