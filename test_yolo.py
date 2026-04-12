"""
YOLO 植物器官检测测试脚本
用于模型验证和推理测试
"""

from ultralytics import YOLO
import os
import cv2
import numpy as np

def validate_model():
    """在测试集上验证模型性能"""
    print("=" * 60)
    print("模型验证")
    print("=" * 60)
    
    # 模型路径
    model_path = 'runs/detect/plant_organ_detection/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在：{model_path}")
        print("请先运行训练脚本 train_yolo.py")
        return
    
    # 加载模型
    print(f"\n加载模型：{model_path}")
    model = YOLO(model_path)
    print("✓ 模型加载完成")
    
    # 在测试集上评估
    print("\n开始在测试集上评估...")
    metrics = model.val(data='data.yaml', split='test')
    
    # 打印结果
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"\n各类别性能:")
    print(f"  Leaf:   {metrics.box.mp[0]:.4f} (precision), {metrics.box.mr[0]:.4f} (recall)")
    print(f"  flower: {metrics.box.mp[1]:.4f} (precision), {metrics.box.mr[1]:.4f} (recall)")
    print(f"  fruit:  {metrics.box.mp[2]:.4f} (precision), {metrics.box.mr[2]:.4f} (recall)")
    print("=" * 60)
    
    return metrics

def predict_image(image_path):
    """单张图像推理"""
    print("=" * 60)
    print(f"图像推理：{image_path}")
    print("=" * 60)
    
    # 模型路径
    model_path = 'runs/detect/plant_organ_detection/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在：{model_path}")
        return
    
    # 加载模型
    model = YOLO(model_path)
    
    # 推理
    print(f"\n推理图像：{image_path}")
    results = model(image_path)
    
    # 显示结果
    result = results[0]
    
    # 打印检测结果
    print(f"\n检测到 {len(result.boxes)} 个目标:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        
        print(f"  - {class_name}: {confidence:.2%} (x1:{bbox[0]:.1f}, y1:{bbox[1]:.1f}, x2:{bbox[2]:.1f}, y2:{bbox[3]:.1f})")
    
    # 保存可视化结果
    output_path = 'prediction_result.jpg'
    result.save(filename=output_path)
    print(f"\n可视化结果已保存：{output_path}")
    
    # 使用 OpenCV 显示（可选）
    img = cv2.imread(output_path)
    if img is not None:
        cv2.imshow('Detection Result', img)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result

def predict_batch(image_folder):
    """批量推理文件夹中的所有图像"""
    print("=" * 60)
    print(f"批量推理：{image_folder}")
    print("=" * 60)
    
    # 模型路径
    model_path = 'runs/detect/plant_organ_detection/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在：{model_path}")
        return
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for f in os.listdir(image_folder) 
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    print(f"\n找到 {len(image_files)} 张图像")
    
    # 批量推理
    output_folder = 'predictions'
    os.makedirs(output_folder, exist_ok=True)
    
    for i, img_name in enumerate(image_files, 1):
        img_path = os.path.join(image_folder, img_name)
        print(f"\n[{i}/{len(image_files)}] 处理：{img_name}")
        
        results = model(img_path)
        result = results[0]
        
        # 保存结果
        output_path = os.path.join(output_folder, f'detected_{img_name}')
        result.save(filename=output_path)
        
        # 打印检测结果
        print(f"  检测到 {len(result.boxes)} 个目标")
    
    print(f"\n✓ 批量推理完成！结果保存在：{output_folder}/")

def export_model():
    """导出模型为 ONNX 格式"""
    print("=" * 60)
    print("模型导出")
    print("=" * 60)
    
    model_path = 'runs/detect/plant_organ_detection/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在：{model_path}")
        return
    
    model = YOLO(model_path)
    
    # 导出为 ONNX
    print("\n导出为 ONNX 格式...")
    model.export(format='onnx')
    print("✓ 导出完成：best.onnx")

if __name__ == '__main__':
    import sys
    
    print("\nYOLO 植物器官检测 - 测试脚本")
    print("=" * 60)
    print("使用方法:")
    print("  1. 验证模型：python test_yolo.py validate")
    print("  2. 单张推理：python test_yolo.py predict <image_path>")
    print("  3. 批量推理：python test_yolo.py batch <image_folder>")
    print("  4. 导出模型：python test_yolo.py export")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        # 默认执行验证
        print("\n未指定命令，执行模型验证...")
        validate_model()
    else:
        command = sys.argv[1]
        
        if command == 'validate':
            validate_model()
        elif command == 'predict' and len(sys.argv) > 2:
            predict_image(sys.argv[2])
        elif command == 'batch' and len(sys.argv) > 2:
            predict_batch(sys.argv[2])
        elif command == 'export':
            export_model()
        else:
            print(f"\n未知命令：{command}")
            print("请使用：validate | predict | batch | export")
