# YOLO 盆栽植物检测

基于 Ultralytics YOLOv8 的室内盆栽植物检测项目，支持树莓派等边缘设备部署。

## 项目概述

本项目包含两个检测任务：

| 任务 | 模型 | 数据集 | 类别 | 用途 |
|------|------|--------|------|------|
| 室内盆栽检测 | YOLOv8n | HomeObjects-3K | potted plant | 树莓派实时检测 |
| 植物器官检测 | YOLOv8n/l | Nature3 | leaf, flower, fruit | 植物器官识别 |

## 项目结构

```
├── train_indoor_plant.py       # 室内盆栽训练脚本（树莓派优化）
├── test_indoor_plant.py        # 室内盆栽测试/推理/导出脚本
├── benchmark_ncnn.py           # PyTorch vs NCNN 速度对比
├── test_camera_ncnn.py         # NCNN 摄像头实时推理
├── train_yolo.py               # 植物器官检测训练脚本
├── test_yolo.py                # 植物器官检测测试脚本
├── square_frame_detector.py    # OpenCV 方框识别（辅助工具）
├── data.yaml                   # 植物器官数据集配置
├── datasets/
│   ├── indoor_potted_plant/    # 室内盆栽数据集
│   └── se00n00/                # Nature3 植物器官数据集
└── runs/
    └── detect/
        └── indoor_potted_plant_pi/
            └── weights/
                ├── best.pt         # PyTorch 模型
                └── best_ncnn_model # NCNN 模型（树莓派）
```

## 快速开始

### 环境安装

```bash
pip install ultralytics opencv-python
```

### 室内盆栽检测（推荐）

**训练模型：**

```bash
python train_indoor_plant.py
```

训练参数：YOLOv8n，图像尺寸 416x416，batch 32（GPU）/ 8（CPU），100 epochs。

**测试集验证：**

```bash
python test_indoor_plant.py validate
```

**单张图像推理：**

```bash
python test_indoor_plant.py predict <图像路径>
```

**摄像头实时检测：**

```bash
python test_indoor_plant.py camera
```

**导出 ONNX / NCNN 格式：**

```bash
python test_indoor_plant.py export
```

### 植物器官检测

```bash
python train_yolo.py
```

数据集：Nature3（28,694 训练 + 3,700 验证 + 2,934 测试），类别：leaf、flower、fruit。

## 树莓派部署

推荐使用 NCNN 格式加速推理：

```python
from ultralytics import YOLO

model = YOLO('runs/detect/indoor_potted_plant_pi/weights/best_ncnn_model')
results = model(frame, imgsz=320, conf=0.5, verbose=False)
```

性能优化建议：
- `imgsz=320`：降低推理延迟
- `conf=0.5`：过滤低置信度结果
- 使用 NCNN 格式替代 PyTorch，ARM 设备上速度更快

## 性能基准

运行 PyTorch vs NCNN 速度对比：

```bash
python benchmark_ncnn.py
```

## 数据集

### 室内盆栽（HomeObjects-3K）

| 集合 | 数量 |
|------|------|
| 训练集 | 1,425 张 |
| 验证集 | 318 张 |
| 测试集 | 159 张 |

### 植物器官（Nature3）

| 集合 | 数量 |
|------|------|
| 训练集 | 28,694 张 |
| 验证集 | 3,700 张 |
| 测试集 | 2,934 张 |

## 依赖

- Python 3.8+
- ultralytics
- opencv-python
- torch
- numpy
