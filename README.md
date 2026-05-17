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

## 树莓派部署经验（重要！）

本项目在 **Raspberry Pi 4 Model B (Cortex-A72, ARMv8-A)** 上的完整移植过程，踩过大量坑，经验总结如下。

### 最终成功方案

使用 **ONNX Runtime** 纯推理，绕过 PyTorch。

```bash
cd ~/YOLO_Learn && source venv/bin/activate
python3 test_camera_onnx.py 0
```

### 踩坑记录（按时间顺序）

| # | 问题 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | `Illegal instruction` | torch 2.12.0 的 aarch64 wheel 编译时使用了 Cortex-A72 不支持的高级 ARM 指令（如 SVE、INT8 矩阵乘法等） | 弃用 PyTorch，改用 ONNX Runtime |
| 2 | NCNN 300 框乱码 | ultralytics 8.4.37 导出的 NCNN 在连续调用时 NMS 后处理失效，第二次推理开始输出 300 个乱框 | 弃用 NCNN，改用 ONNX Runtime |
| 3 | `cv2.imshow` 弹不出窗口 | SSH 会话中 `DISPLAY` 未设置；pip 版 `opencv-python` 在 aarch64 上缺少 GTK/Qt GUI 后端 | 在脚本开头设置 `os.environ["DISPLAY"] = ":0"` |
| 4 | `pip install opencv-python` 太慢/失败 | PyPI 上 aarch64 的 `opencv-python` wheel 体积大（46MB），且容易安装系统版 `python3-opencv` 冲突 | 保持现有环境，不折腾 OpenCV |
| 5 | venv 被误删 | 文档中写了 `rm -rf venv`，用户执行后丢失所有已装包 | **文档已修正**，不再建议删除 venv |
| 6 | ONNX 后处理 shape 错误 | 单类模型输出为 `(1, 5, N)` 而非 `(1, 6, N)`，没有单独的分类得分列 | 后处理直接取第 5 列作为置信度 |
| 7 | `out[0]` 被取了两次 | `detect()` 里已取 `sess.run(...)[0]`，`_postprocess` 又取一次导致 2D 数组无法 `transpose` | 改为 `out[0].T` |

### 关键代码片段（树莓派专用）

```python
import os
os.environ.setdefault("DISPLAY", ":0")  # 必须，否则 cv2.imshow 无效

import onnxruntime as ort

# 优化后的 ONNX Session
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4

sess = ort.InferenceSession("best.onnx", opts, providers=["CPUExecutionProvider"])

# 单类模型输出: (1, 5, N) -> cx, cy, w, h, conf
# 后处理注意：没有第 6 列的分类得分！
preds = out[0].T  # (N, 5)
scores = preds[:, 4]  # 直接取置信度
```

### 树莓派 4 实测性能

| 指标 | 数值 |
|------|------|
| 推理耗时 | **~230ms/帧** |
| FPS | **~4.1** |
| 检测精度 | 置信度 26%-66%（对准盆栽时稳定检出） |
| 内存占用 | 模型 12MB + ONNX Runtime ~100MB |

### 依赖安装（树莓派 4）

```bash
# 创建 venv（带 --system-site-packages 可以访问系统版 opencv）
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 安装 torch 但不装 CUDA 依赖（--no-deps 跳过 nvidia_cublas 等）
pip install torch torchvision --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

# ultralytics（仅用于训练/导出，推理时不需要）
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

# ONNX Runtime（推理核心）
pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 为什么不用 PyTorch / NCNN？

| 方案 | 可行性 | 原因 |
|------|--------|------|
| PyTorch `best.pt` | ❌ | Illegal instruction，Cortex-A72 不支持 torch 2.12.0 的编译指令 |
| NCNN `best_ncnn_model` | ❌ | ultralytics NCNN 导出有 bug，连续推理 NMS 失效，300 乱框 |
| ONNX `best.onnx` | ✅ | ONNX Runtime 有正式 aarch64 wheel，推理稳定 ~230ms |

## 性能基准

运行 PyTorch vs NCNN 速度对比（PC 上）：

```bash
python benchmark_ncnn.py
```

树莓派上请直接运行：

```bash
python test_camera_onnx.py 0
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
