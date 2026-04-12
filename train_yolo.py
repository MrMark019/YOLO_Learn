"""
YOLO 植物器官检测训练脚本
数据集：Nature3 - Leaf, Flower, and Fruit Detection
类别：Leaf (叶子), flower (花朵), fruit (果实)
"""

from ultralytics import YOLO
import os

def train_yolo():
    """训练 YOLO 模型"""
    
    print("=" * 60)
    print("开始 YOLO 植物器官检测训练")
    print("=" * 60)
    
    # 1. 加载预训练模型（使用 YOLOv8n，最小最快）
    print("\n[1/4] 加载预训练模型 YOLOv8n...")
    model = YOLO('yolov8n.pt')
    print("✓ 模型加载完成")
    
    # 2. 训练配置
    print("\n[2/4] 配置训练参数...")
    training_config = {
        'data': 'data.yaml',           # 数据集配置文件
        'epochs': 100,                 # 训练轮数
        'batch': 16,                   # 批次大小（根据显存调整）
        'imgsz': 640,                  # 输入图像尺寸
        'device': 0,                   # GPU 设备（0 表示第一个 GPU，cpu 表示 CPU）
        'workers': 4,                  # 数据加载线程数
        'optimizer': 'SGD',            # 优化器
        'lr0': 0.01,                   # 初始学习率
        'patience': 50,                # 早停耐心值
        'save': True,                  # 保存模型
        'project': 'runs/detect',      # 项目目录
        'name': 'plant_organ_detection',  # 实验名称
        'exist_ok': False,             # 是否覆盖已有实验
        'verbose': True,               # 详细输出
        'pretrained': True,            # 使用预训练权重
    }
    
    # 打印配置信息
    print("\n训练配置:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    # 3. 开始训练
    print("\n[3/4] 开始训练...")
    print("训练过程可能需要 2-6 小时，请耐心等待...")
    print("\n监控指标:")
    print("  - mAP@50: 目标 > 0.85")
    print("  - mAP@50-95: 目标 > 0.60")
    print("  - loss: 观察是否收敛\n")
    
    results = model.train(**training_config)
    
    # 4. 训练完成
    print("\n[4/4] 训练完成！")
    print("=" * 60)
    print("训练结果已保存到:")
    print("  runs/detect/plant_organ_detection/")
    print("\n重要文件:")
    print("  - weights/best.pt  ← 最佳模型权重")
    print("  - weights/last.pt  ← 最后一次权重")
    print("  - results.png      ← 训练曲线")
    print("  - confusion_matrix.png ← 混淆矩阵")
    print("=" * 60)
    
    return results

def validate_model():
    """验证训练好的模型"""
    print("\n开始模型验证...")
    
    # 加载最佳模型
    model_path = 'runs/detect/plant_organ_detection/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 在验证集上评估
    metrics = model.val()
    
    print("\n验证结果:")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    
    return metrics

if __name__ == '__main__':
    # 开始训练
    train_results = train_yolo()
    
    # 可选：验证模型
    # validate_model()
