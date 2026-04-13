# YOLO 植物器官检测 - 快速开始指南

## 📦 环境准备

### 1. 安装依赖

```bash
pip install ultralytics
pip install opencv-python
```

### 2. 检查环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import torch; print(f'CUDA 可用：{torch.cuda.is_available()}')"
```

---

## 🚀 快速开始

### 步骤 1: 训练模型

```bash
python train_yolo.py
```

**训练说明：**
- 预计时间：2-6 小时（取决于 GPU）
- 输出目录：`runs/detect/plant_organ_detection/`
- 最佳模型：`runs/detect/plant_organ_detection/weights/best.pt`

### 步骤 2: 验证模型

```bash
python test_yolo.py validate
```

**预期结果：**
- mAP@50: > 0.85
- mAP@50-95: > 0.60

### 步骤 3: 推理测试

**单张图像推理：**
```bash
python test_yolo.py predict <图像路径>
# 例如：
python test_yolo.py predict test_image.jpg
```

**批量推理：**
```bash
python test_yolo.py batch <图像文件夹路径>
# 例如：
python test_yolo.py batch datasets/se00n00/nature3-leaf-flower-and-fruit-detection/versions/1/test/images
```

---

## 📊 训练参数调整

如果显存不足，修改 `train_yolo.py` 中的参数：

```python
training_config = {
    'batch': 8,      # 减小批次大小（默认 128）
    'imgsz': 416,    # 减小图像尺寸（默认 640）
    # ...
}
```

---

## 📁 输出文件说明

训练完成后，在 `runs/detect/plant_organ_detection/` 目录：

```
runs/detect/plant_organ_detection/
├── weights/
│   ├── best.pt          ← 最佳模型权重
│   └── last.pt          ← 最后一次迭代权重
├── results.png          ← 训练曲线（loss、mAP）
├── confusion_matrix.png ← 混淆矩阵
├── PR_curve.png         ← 精确率 - 召回率曲线
├── F1_curve.png         ← F1 分数曲线
├── labels_correlogram.png
└── args.yaml            ← 训练参数配置
```

---

## 🔧 常见问题

### Q1: CUDA 内存不足
**解决：** 减小 `batch` 和 `imgsz` 参数

### Q2: 训练速度慢
**解决：** 
- 确保使用 GPU（`device=0`）
- 增加 `workers` 数量（默认 4）

### Q3: mAP 太低
**解决：**
- 增加训练轮数（`epochs`）
- 尝试更大的模型（`yolov8s.pt`, `yolov8m.pt`）
- 检查数据集质量

---

## 🎯 使用训练好的模型

### Python 代码推理

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/plant_organ_detection/weights/best.pt')

# 推理
results = model('test_image.jpg')

# 处理结果
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        
        print(f"检测到：{class_name}, 置信度：{confidence:.2%}")
        print(f"边界框：{bbox}")
```

---

## 📈 性能优化

### 导出为 ONNX（加速推理）

```bash
python test_yolo.py export
```

### 使用更大的模型

修改 `train_yolo.py`：
```python
model = YOLO('yolov8s.pt')  # 小型
model = YOLO('yolov8m.pt')  # 中型
model = YOLO('yolov8l.pt')  # 大型
```

---

## 📝 下一步

1. ✅ 训练模型
2. ✅ 验证性能
3. ⬜ 测试推理
4. ⬜ 集成到你的项目
5. ⬜ 部署应用

---

**祝你训练顺利！** 🎉
