# 室内盆栽植物检测 - 快速开始

## 项目背景

从 `HomeObjects-3K` 数据集中提取了 `potted plant` 类别，构建了一个专门用于**室内盆栽检测**的数据集。目标是在树莓派等边缘设备上快速识别并定位室内盆栽植物。

## 数据集信息

| 集合 | 图片数量 | 说明 |
|------|---------|------|
| 训练集 (train) | 1,425 张 | 室内场景中的盆栽植物 |
| 验证集 (val) | 318 张 | 模型调参用 |
| 测试集 (test) | 159 张 | 最终评估用 |
| **总计** | **1,902 张** | 单类别：potted plant |

数据集路径：`datasets/indoor_potted_plant/`

## 文件说明

| 文件 | 用途 |
|------|------|
| `train_indoor_plant.py` | 训练脚本（YOLOv8n，树莓派优化参数） |
| `test_indoor_plant.py` | 测试/推理/导出脚本 |
| `datasets/indoor_potted_plant/data.yaml` | 数据集配置 |

## 快速开始

### 1. 训练模型

```bash
python train_indoor_plant.py
```

**训练参数：**
- 模型：`yolov8n.pt`（专为边缘设备优化的轻量模型）
- 图像尺寸：`416×416`
- 批次大小：`32`（GPU）/ `8`（CPU）
- 输出目录：`runs/detect/indoor_potted_plant_pi/`

### 2. 测试集验证

```bash
python test_indoor_plant.py validate
```

### 3. 单张图像推理

```bash
python test_indoor_plant.py predict testing_pic/image.png
```

### 4. 摄像头实时检测（树莓派）

```bash
python test_indoor_plant.py camera
```

> 按 `q` 键退出实时检测窗口。

### 5. 导出树莓派加速格式

```bash
python test_indoor_plant.py export
```

这会同时导出：
- **ONNX** 格式：通用加速格式
- **NCNN** 格式：ARM/树莓派上性能更好

## 树莓派部署建议

### 推理参数优化

在树莓派上运行时，强烈建议使用以下参数：

```python
from ultralytics import YOLO

model = YOLO('runs/detect/indoor_potted_plant_pi/weights/best.pt')
results = model(frame, imgsz=320, conf=0.5, verbose=False)
```

- `imgsz=320`：大幅降低推理延迟
- `conf=0.5`：过滤低置信度结果
- `verbose=False`：减少日志输出

### NCNN 部署（极致性能）

```python
from ultralytics import YOLO

model = YOLO('runs/detect/indoor_potted_plant_pi/weights/best_ncnn_model')
results = model(frame, imgsz=320)
```

## 输出文件

训练完成后，在 `runs/detect/indoor_potted_plant_pi/` 目录下：

```
weights/
├── best.pt          ← 最佳模型
├── last.pt          ← 最后模型
├── best.onnx        ← ONNX 格式（导出后）
└── best_ncnn_model/ ← NCNN 格式（导出后）
```

## 下一步

1. ✅ 运行 `python train_indoor_plant.py` 训练模型
2. ✅ 用 `test_indoor_plant.py` 在 PC 上验证效果
3. ⬜ 将 `best.pt` 或 NCNN 模型复制到树莓派
4. ⬜ 在树莓派上接摄像头运行实时检测
