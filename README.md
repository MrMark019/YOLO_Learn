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

---

## How to develop（开发指南）

本节记录本项目的完整开发经验、方法论、踩坑教训，以及当前实现进度和未来任务规划。

### 1. 项目架构与数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                         系统总体架构                                 │
├─────────────────────────────────────────────────────────────────────┤
│  PC端 (训练/开发)        树莓派4 (边缘推理)         ESP32 (控制中枢)  │
│  ───────────────        ────────────────         ─────────────────  │
│  ┌──────────┐           ┌─────────────┐          ┌──────────────┐  │
│  │YOLOv8训练 │──best.onnx──▶│ONNX Runtime │──UART──▶│ 串口接收/控制 │  │
│  │(Ultralytics│           │摄像头+后处理│ 115200   │ 小车运动决策 │  │
│  │  GPU服务器)│           │/dev/serial0│          │ GPIO驱动电机 │  │
│  └──────────┘           └─────────────┘          └──────────────┘  │
│                              ▲                                      │
│                              │ START/STOP 控制命令                   │
│                         ┌────┴────┐                                 │
│                         │ 串口监听 │                                  │
│                         │systemd │                                  │
│                         │服务自启│                                  │
│                         └─────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. 开发方法论

#### 2.1 模型选型原则

边缘部署必须优先考虑**模型大小**和**推理速度**，而非极限精度：

| 场景 | 推荐模型 | 参数量 | 大小 | 推理速度(Pi4) |
|------|----------|--------|------|--------------|
| 边缘实时检测 | YOLOv8n | 3.2M | 6MB | ~230ms/帧 |
| 精度优先云端 | YOLOv8x | 68.2M | 131MB | 不适用 |

**经验**：本项目从YOLOv8n训练到ONNX导出，全程保持模型在6MB以内，确保树莓派加载无压力。

#### 2.2 推理框架选择决策树

树莓派ARM平台部署时，按以下优先级评估推理框架：

```
是否需要PyTorch生态功能？
  ├─ 是 → PyTorch (但注意ARM指令集兼容性!)
  └─ 否 → 纯推理场景
       ├─ 需要GPU加速？ → TensorRT / MNN
       └─ 纯CPU推理？
            ├─ 追求极致速度且能接受量化？ → NCNN / MNN
            └─ 追求稳定性、兼容性？ → ONNX Runtime ✅ 推荐
```

**本项目最终选择ONNX Runtime的原因**：
- PyTorch 2.12.0 aarch64 wheel 使用了Cortex-A72不支持的SVE/INT8指令
- Ultralytics 8.4.37 NCNN导出存在连续推理NMS失效的Bug
- ONNX Runtime有官方aarch64 wheel，推理稳定，后处理完全可控

#### 2.3 数据集处理流水线

从原始数据集到训练集的标准流程：

```
原始数据集(如HomeObjects-3K)
    │
    ▼
[步骤1] 按类别ID筛选目标类别 (potted plant = class 10)
    │
    ▼
[步骤2] 重映射类别ID为0 (YOLO单类检测)
    │
    ▼
[步骤3] 划分训练/验证/测试集 (建议 8:1.5:0.5 或 75:15:10)
    │
    ▼
[步骤4] 生成 data.yaml 配置文件
    │
    ▼
[步骤5] Ultralytics训练 → 导出ONNX
```

**关键注意点**：
- 单类检测模型的ONNX输出为 `(1, 5, N)`，5=xywh+conf，**没有分类得分列**
- 多类检测模型的ONNX输出为 `(1, 5+num_classes, N)`，需要额外处理分类分支
- 后处理时必须先确认输出维度，否则会出现shape错误

#### 2.4 后处理实现规范

标准的YOLO后处理流程（本项目ONNX Runtime实现）：

```python
# 1. 模型输出: (1, 5, 3549) for single-class
preds = out[0].T                      # (3549, 5)

# 2. 置信度过滤
mask = preds[:, 4] > CONF             # 第5列是置信度
filtered = preds[mask]

# 3. 坐标解码 (模型输入416x416 → 原始图像尺寸)
sx, sy = orig_w / 416, orig_h / 416
x1 = (cx - w/2) * sx
y1 = (cy - h/2) * sy
x2 = (cx + w/2) * sx
y2 = (cy + h/2) * sy

# 4. NMS去重
idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF, IOU)
```

**常见Bug**：
- ❌ `np.transpose(out[0], (0,2,1))` — 如果`out[0]`已经是2D，axes参数会越界
- ✅ `out[0].T` — 直接转置，最安全
- ❌ 使用`ultralytics`的默认后处理 — 对单类模型会产生shape不匹配

### 3. 树莓派部署核心注意事项

#### 3.1 串口配置（极易踩坑）

树莓派GPIO UART默认被系统控制台占用，必须**完全禁用串口登录**才能给程序使用：

```bash
# 1. 禁用串口控制台登录（必须从cmdline.txt移除console参数）
sudo sed -i 's/console=serial0,115200 //' /boot/cmdline.txt

# 2. 确保硬件串口已启用
grep "enable_uart" /boot/firmware/config.txt  # 应为 enable_uart=1

# 3. 禁用串口登录服务
sudo systemctl stop serial-getty@ttyS0.service
sudo systemctl disable serial-getty@ttyS0.service

# 4. 重启生效
sudo reboot
```

**验证串口是否释放**：
```bash
ls -la /dev/serial0   # 应存在且指向ttyS0
ls -la /dev/ttyS0     # 权限应为 crw-rw---- root dialout
```

**若串口设备不存在**：检查`/boot/firmware/config.txt`中的`enable_uart=0`被错误设置，改为`enable_uart=1`后重启。

#### 3.2 显示环境检测

树莓派可能无外接显示器，程序必须能自动适应：

```python
def check_display():
    display = os.environ.get("DISPLAY", "")
    return bool(display)

# 无显示器时跳过cv2.imshow，避免程序卡住
if not has_display:
    print("[INFO] 未检测到显示环境，以headless模式运行")
```

#### 3.3 systemd服务配置

实现开机自启和崩溃自动重启：

```ini
[Unit]
Description=YOLO Detection Service
After=network.target

[Service]
Type=simple
User=mark
WorkingDirectory=/home/mark/YOLO_Learn
ExecStart=/home/mark/YOLO_Learn/venv/bin/python3 /home/mark/YOLO_Learn/test_camera_onnx_headless.py
Restart=always
RestartSec=5
Environment="DISPLAY=:0"

[Install]
WantedBy=multi-user.target
```

**管理命令**：
```bash
sudo systemctl start yolo-detection.service   # 启动
sudo systemctl stop yolo-detection.service    # 停止
sudo systemctl status yolo-detection.service  # 查看状态
journalctl -u yolo-detection.service -f       # 实时查看日志
```

#### 3.4 虚拟环境管理

**绝对不要删除已有的venv！** 树莓派上编译安装包非常慢，且可能破坏系统Python环境。

```bash
# 推荐做法：使用 --system-site-packages 继承系统opencv
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 安装ONNX Runtime（不依赖torch）
pip install onnxruntime

# 如需torch（仅训练/导出用），用 --no-deps 跳过CUDA依赖
pip install torch torchvision --no-deps
```

### 4. 串口通信协议设计

#### 4.1 协议格式

本项目采用纯文本协议，便于调试：

```
# 树莓派 → ESP32 (检测数据)
F=帧号,P=盆栽数量,B=x1,y1,x2,y2,置信度;...

示例：
F=676,P=1,B=119,-2,643,479,0.37
F=100,P=2,B=100,50,300,400,0.75;350,80,600,450,0.62

# ESP32 → 树莓派 (控制命令)
START\n   # 开始检测
STOP\n    # 暂停检测

# 树莓派 → ESP32 (状态回执)
RUNNING\n  # 已进入检测状态
PAUSED\n   # 已暂停检测
```

#### 4.2 通信时序

```
ESP32          树莓派
  │               │
  │── START ────▶│  触发检测
  │               │
  │◀── RUNNING ──│  确认启动
  │               │
  │◀── F=1,P=0 ──│  持续发送检测数据...
  │◀── F=2,P=1 ──│
  │               │
  │── STOP ─────▶│  暂停检测
  │               │
  │◀── PAUSED ───│  确认暂停
```

#### 4.3 非阻塞读取实现

```python
def read_serial_command(ser):
    """非阻塞读取串口命令"""
    if not ser or not ser.is_open:
        return None
    cmd = None
    while ser.in_waiting > 0:           # 有数据才读
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "START":
            cmd = "START"
        elif line == "STOP":
            cmd = "STOP"
    return cmd
```

### 5. ESP32开发注意事项

#### 5.1 接线（必须共地）

```
树莓派 GPIO14(TXD, Pin8)  →  ESP32 RX (GPIO16)
树莓派 GPIO15(RXD, Pin10) →  ESP32 TX (GPIO17)
树莓派 GND (Pin6)         →  ESP32 GND

波特率: 115200, 8N1
```

**关键**：GND必须连接！否则电平参考不一致导致乱码。

#### 5.2 Arduino代码结构

ESP32使用`Serial2`（UART2）与树莓派通信，`Serial`（USB串口）用于调试：

```cpp
Serial.begin(115200);                          // USB调试
Serial2.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);  // 树莓派连接
```

状态检测使用**边沿触发**（检测变化而非电平），避免连续发送：

```cpp
if (runDetection != prevState) {   // 状态变化时才发送
    prevState = runDetection;
    Serial2.print(runDetection ? "START\n" : "STOP\n");
}
```

### 6. SSH远程开发工作流

本项目使用paramiko进行SSH远程管理：

```python
# ssh_pi.py - 执行远程命令
python ssh_pi.py "ls -la"           # 单条命令
python ssh_pi.py                    # 交互式shell

# upload_to_pi.py - 上传文件
python upload_to_pi.py local.py /home/mark/YOLO_Learn/local.py
```

**文件同步策略**：
1. 本地修改代码 → 测试通过后 → 上传到树莓派
2. 树莓派上验证运行 → 查看日志确认正常
3. 使用systemd服务管理长期运行进程

### 7. 调试技巧

#### 7.1 树莓派日志查看

```bash
# 实时跟踪服务日志
sudo journalctl -u yolo-detection.service -f

# 查看最近50行
sudo journalctl -u yolo-detection.service --no-pager -n 50

# 查看所有日志（包括历史）
sudo journalctl -u yolo-detection.service --no-pager
```

#### 7.2 串口调试

```bash
# 在树莓派上监听串口（临时测试用）
stty -F /dev/serial0 115200 raw
cat /dev/serial0          # 查看接收到的数据
echo "STOP" > /dev/serial0  # 手动发送命令
```

#### 7.3 ONNX模型验证

```python
import onnxruntime as ort

sess = ort.InferenceSession("best.onnx")
# 打印输入输出形状
print("Inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
print("Outputs:", [(o.name, o.shape) for o in sess.get_outputs()])

# 单类模型输出应为: [1, 5, 3549]
# 多类模型输出应为: [1, 5+num_classes, 3549]
```

### 8. 项目当前实现进度

#### 8.1 已完成功能 ✅

| 模块 | 功能 | 状态 | 文件 |
|------|------|------|------|
| 数据集构建 | HomeObjects-3K提取potted plant，划分train/val/test | ✅ | `datasets/indoor_potted_plant/` |
| 模型训练 | YOLOv8n训练，mAP@50=75.17% | ✅ | `train_indoor_plant.py` |
| PC端推理 | PyTorch摄像头实时检测，~14.8 FPS | ✅ | `test_indoor_plant.py` |
| ONNX导出 | 导出best.onnx（11.6MB） | ✅ | 通过Ultralytics导出 |
| 树莓派部署 | ONNX Runtime推理，~4.5 FPS | ✅ | `test_camera_onnx.py` |
| 无GUI模式 | headless运行，自动跳过cv2.imshow | ✅ | `test_camera_onnx_headless.py` |
| ESP32控制 | START/STOP命令发送，三触发方式 | ✅ | `esp32_controller.ino` |
| 串口通信 | 树莓派↔ESP32双向通信，协议已验证 | ✅ | 串口协议已跑通 |
| systemd服务 | 开机自启，崩溃自动重启 | ✅ | `yolo-detection.service` |
| 自动保存 | 检测到目标时自动截图 | ✅ | `camera_captures/` |
| SSH工具 | 远程命令执行、文件上传 | ✅ | `ssh_pi.py`, `upload_to_pi.py` |

#### 8.2 进行中/待完善 ⚠️

| 模块 | 功能 | 状态 | 说明 |
|------|------|------|------|
| 小车运动控制 | 根据检测坐标驱动电机 | ⏳ | ESP32端已预留控制逻辑入口，待实现 |
| 多盆栽路径规划 | 检测到多个盆栽时的巡航策略 | ⏳ | 需设计优先级算法 |
| 通信可靠性 | 串口校验和、重传机制 | ⏳ | 当前为纯文本无校验，抗干扰能力有限 |
| 异常恢复 | 摄像头断开、模型加载失败自动恢复 | ⏳ | 当前需要手动重启服务 |

#### 8.3 未来扩展方向 📋

| 方向 | 描述 | 优先级 |
|------|------|--------|
| 模型量化 | INT8量化加速，目标提升到8-10 FPS | 高 |
| 数据集扩充 | 增加光照/角度/遮挡多样性 | 高 |
| 多任务扩展 | 增加植物健康状态分类（健康/缺水/病害） | 中 |
| 云端对接 | MQTT协议连接云平台，远程监控 | 中 |
| SLAM导航 | 小车自主建图与路径规划 | 低 |

### 9. 核心踩坑总结

| # | 问题 | 根本原因 | 解决方案 |
|---|------|----------|----------|
| 1 | PyTorch Illegal Instruction | torch 2.12.0 aarch64使用Cortex-A72不支持的ARM指令 | 改用ONNX Runtime |
| 2 | NCNN 300乱框 | Ultralytics 8.4.37 NCNN导出Bug，连续推理NMS失效 | 改用ONNX Runtime |
| 3 | cv2.imshow无窗口 | SSH会话无DISPLAY环境变量 | 设置`os.environ["DISPLAY"] = ":0"` |
| 4 | pip安装磁盘空间不足 | PyPI torch拉取CUDA依赖(nvidia_cublas等) | 使用`--no-deps`安装 |
| 5 | ONNX后处理shape错误 | 单类模型输出(1,5,N)，代码按(1,6,N)处理 | 改为`out[0].T`，取第5列作为置信度 |
| 6 | `out[0]`被取两次 | detect()已取`sess.run()[0]`，postprocess又取一次 | 直接使用`out[0].T` |
| 7 | 串口被系统占用 | ttyS0默认开启串口控制台登录 | 从cmdline.txt移除console参数，禁用getty服务 |
| 8 | 串口设备消失 | raspi-config异常退出导致enable_uart=0 | 手动修改/boot/firmware/config.txt为enable_uart=1 |
| 9 | ONNX首帧51秒 | Ultralytics加载ONNX有额外开销 | 使用纯ONNX Runtime，绕过Ultralytics推理管线 |
| 10 | venv被误删 | 文档建议rm -rf venv | 修正文档，改用覆盖安装而非删除 |

### 10. 性能基准参考

| 平台 | 推理框架 | 输入尺寸 | 单帧耗时 | FPS | 备注 |
|------|----------|----------|----------|-----|------|
| PC (i7-13700K) | PyTorch | 416x416 | ~68ms | ~14.8 | CUDA加速 |
| PC (i7-13700K) | ONNX Runtime | 416x416 | ~45ms | ~22 | CPU推理 |
| 树莓派4 | ONNX Runtime | 416x416 | ~230ms | ~4.5 | 4线程优化 |
| 树莓派4 | PyTorch | 416x416 | 崩溃 | 0 | Illegal Instruction |
| 树莓派4 | NCNN | 416x416 | 首帧正常，后续300框 | 不可用 | NMS Bug |

---

## 附录：文件清单与用途速查

| 文件 | 用途 | 运行平台 |
|------|------|----------|
| `train_indoor_plant.py` | 训练脚本 | PC/GPU服务器 |
| `test_indoor_plant.py` | PC端推理/测试/导出 | PC |
| `test_camera_onnx.py` | 树莓派摄像头检测（带GUI） | 树莓派 |
| `test_camera_onnx_headless.py` | 树莓派摄像头检测（无GUI，支持START/STOP） | 树莓派 |
| `benchmark_ncnn.py` | PyTorch vs NCNN速度对比 | PC |
| `esp32_controller.ino` | ESP32控制器（发送START/STOP） | ESP32 |
| `esp32_serial_receiver.ino` | ESP32接收器（解析检测数据） | ESP32 |
| `ssh_pi.py` | SSH远程命令执行 | PC |
| `upload_to_pi.py` | 文件上传到树莓派 | PC |
| `package_for_pi.py` | 打包部署文件 | PC |
| `yolo-detection.service` | systemd服务配置 | 树莓派 |
