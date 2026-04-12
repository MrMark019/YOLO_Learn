# YOLO 植物器官检测训练计划

## 📋 项目概述

使用 Nature3 数据集训练 YOLO 模型，检测植物器官（叶子、花朵、果实）的位置。

### 数据集信息
- **数据集名称**: Nature3: Leaf, Flower, and Fruit Detection
- **数据集路径**: `d:\MarkLab\YOLO_Learn\datasets\se00n00\nature3-leaf-flower-and-fruit-detection\versions\1`
- **类别数**: 3 类
  - `0: Leaf` (叶子)
  - `1: flower` (花朵)
  - `2: fruit` (果实)
- **数据规模**:
  - 训练集: 28,694 张图像
  - 验证集: 3,700 张图像
  - 测试集: 2,934 张图像

---

## 🎯 训练目标

1. 训练一个能够检测叶子、花朵、果实位置的 YOLO 模型
2. 输出每个检测目标的类别和边界框坐标
3. 可以用于后续的花盆位置推断

---

## 🛠️ 实现步骤

### 步骤 1: 环境准备
- 安装 ultralytics 库（YOLOv8/v11）
- 检查 GPU 可用性
- 验证数据集路径和格式

### 步骤 2: 数据集配置
- 创建/验证 `data.yaml` 配置文件
- 检查数据集结构是否符合 YOLO 要求
- 验证图像和标注文件对应关系

### 步骤 3: 模型选择
- 选择预训练模型（推荐 YOLOv8n 或 YOLOv11n）
- 使用 COCO 预训练权重进行迁移学习
- 模型对比：
  - `yolov8n.pt` / `yolo11n.pt`: 最小最快，适合快速测试
  - `yolov8s.pt` / `yolo11s.pt`: 小型，平衡速度和精度
  - `yolov8m.pt` / `yolo11m.pt`: 中型，精度更高

### 步骤 4: 训练配置
- 设置训练参数：
  - `epochs`: 100（初始训练）
  - `batch`: 16（根据显存调整）
  - `imgsz`: 640（标准输入尺寸）
  - `device`: 0（GPU）或 cpu
  - `workers`: 4（数据加载线程数）
  - `optimizer`: SGD 或 AdamW
  - `lr0`: 0.01（初始学习率）

### 步骤 5: 执行训练
- 启动训练过程
- 监控训练指标（loss、mAP）
- 保存最佳模型权重

### 步骤 6: 模型验证
- 在测试集上评估模型性能
- 查看 mAP@50 和 mAP@50-95 指标
- 分析混淆矩阵和 PR 曲线

### 步骤 7: 推理测试
- 使用训练好的模型进行预测
- 可视化检测结果
- 输出边界框坐标和类别信息

### 步骤 8: 导出模型（可选）
- 导出为 ONNX 格式
- 导出为 TensorRT 格式（部署优化）

---

## 📁 项目文件结构

```
d:\MarkLab\YOLO_Learn\
├── YOLO_Training_Plan.md          ← 本规划文档
├── download_flower_dataset.py     ← 数据集下载脚本
├── train_yolo.py                  ← 训练脚本（待创建）
├── test_yolo.py                   ← 测试/推理脚本（待创建）
├── data.yaml                      ← 数据集配置（待创建/复制）
├── datasets/
│   └── se00n00/
│       └── nature3-leaf-flower-and-fruit-detection/
│           └── versions/
│               └── 1/
│                   ├── data.yaml
│                   ├── train/
│                   │   ├── images/
│                   │   └── labels/
│                   ├── valid/
│                   │   ├── images/
│                   │   └── labels/
│                   └── test/
│                       ├── images/
│                       └── labels/
└── runs/
    └── detect/
        └── train/                 ← 训练输出目录
            ├── weights/
            │   ├── best.pt        ← 最佳模型权重
            │   └── last.pt        ← 最后一次权重
            ├── results.png        ← 训练曲线
            ├── confusion_matrix.png
            └── ...
```

---

## 💻 核心代码实现

### 1. 训练脚本 (train_yolo.py)
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练模型
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    optimizer='SGD',
    lr0=0.01,
    patience=50,
    save=True,
    project='runs/detect',
    name='plant_organ_detection'
)
```

### 2. 测试脚本 (test_yolo.py)
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/plant_organ_detection/weights/best.pt')

# 在测试集上评估
metrics = model.val()

# 单张图像推理
results = model('test_image.jpg')
results[0].show()
```

---

## 📊 预期输出

### 训练指标
- **mAP@50**: 目标 > 0.85
- **mAP@50-95**: 目标 > 0.60
- **训练时间**: 约 2-6 小时（取决于 GPU）

### 输出文件
- `best.pt`: 最佳模型权重
- `results.png`: 训练曲线图
- `confusion_matrix.png`: 混淆矩阵
- `PR_curve.png`: 精确率-召回率曲线
- `F1_curve.png`: F1 分数曲线

---

## ⚠️ 注意事项

1. **显存要求**:
   - YOLOv8n: 至少 4GB GPU 显存
   - 如果显存不足，减小 batch size

2. **训练时间**:
   - CPU 训练会非常慢（不推荐）
   - GPU 训练约 2-6 小时

3. **数据验证**:
   - 训练前验证数据集完整性
   - 检查图像和标注文件是否一一对应

4. **过拟合监控**:
   - 观察训练 loss 和验证 loss
   - 使用 early stopping（patience=50）

---

## 🚀 后续优化方向

1. **数据增强**: 调整增强策略提高泛化能力
2. **超参数调优**: 使用 `model.tune()` 自动调参
3. **模型融合**: 尝试更大的模型（YOLOv8m/l/x）
4. **部署优化**: 导出为 ONNX/TensorRT 加速推理

---

## 📝 执行顺序

1. ✅ 下载数据集（已完成）
2. ⬜ 创建 data.yaml 配置文件
3. ⬜ 创建训练脚本 train_yolo.py
4. ⬜ 执行训练
5. ⬜ 验证模型性能
6. ⬜ 创建推理脚本 test_yolo.py
7. ⬜ 测试推理效果

---

**准备就绪后，开始实现！**
