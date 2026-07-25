# YOLO 盆栽植物检测 — 树莓派 Web 控制器

基于 YOLOv8 ONNX Runtime 的盆栽植物实时检测系统，部署在 Raspberry Pi 4B 上，通过 Web 面板远程控制，与 ESP32 串口联动实现小车/摄像头控制。

## 系统架构

```
PC浏览器 ←→ WiFi ←→ 树莓派 (Flask Web :5000 + MJPEG :5001 + ONNX 检测)
                       ↕ /dev/serial0 (UART 115200)
                    ESP32 (小车驱动 + 摄像头云台)
```

## 项目结构

```
YOLO_Learn/
├── test_camera_onnx_headless.py  # 主检测程序（ONNX Runtime）
├── test_camera_onnx.py           # 带 GUI 的检测程序
├── train_indoor_plant.py         # 室内盆栽训练
├── train_yolo.py                 # 植物器官检测训练
├── test_indoor_plant.py          # PC 端推理/测试/导出
├── test_yolo.py                  # 植物器官测试
├── benchmark_ncnn.py             # 推理框架速度对比
├── ssh_pi.py                     # SSH 工具（WIN 热点）
├── ssh_pi_dhu.py                 # SSH 工具（DHU-1X 校园网）
├── upload_to_pi.py               # 文件上传到树莓派
├── web_server/                   # Web 控制面板
│   ├── app.py                    # Flask + SocketIO 主程序
│   ├── yolo_manager.py           # YOLO 子进程管理器
│   ├── diagnostics.py            # 硬件自检模块
│   ├── mock_yolo.py              # PC 端模拟测试
│   ├── templates/index.html      # 网页界面
│   └── static/
│       ├── css/style.css         # 样式
│       └── js/controller.js      # 前端逻辑
├── esp32_controller.ino          # ESP32 控制端固件
├── esp32_serial_receiver.ino     # ESP32 接收端固件
├── runs/detect/
│   └── indoor_potted_plant_pi/
│       └── weights/best.onnx     # 推理模型（12MB）
└── datasets/                     # 训练数据
```

## Web 控制面板

运行在树莓派端口 `5000`，同一 WiFi 下的设备浏览器可直接访问 `http://<树莓派IP>:5000`。

### 功能

| 模块 | 说明 |
|------|------|
| 小车方向键 | WASD + Q/E 旋转（左逆时针/右顺时针），松手自动发 `x` 停止 |
| 摄像头方向键 | 8=上 2=下 4=左 6=右 |
| 识别控制 | 启动/停止 YOLO 检测按钮 |
| MJPEG 视频流 | 端口 5001，画面上叠加 FPS/帧号/植物数 |
| 硬件自检 | 点击"硬件自检"或启动识别时自动检测欠压/降频/温度 |
| 调试日志 | 实时显示 YOLO 输出、串口数据 `[RX]` |
| 检测坐标 | 实时显示识别目标的边界框坐标和置信度 |
| 植物卡片 | 按键盘 `1` 切换显示绿萝/多肉植物信息 |

### 快捷键

| 按键 | 功能 |
|------|------|
| `1` | 切换植物信息显示 |
| 空格 | 启动/停止 YOLO 识别 |
| W/A/S/D | 小车前进/左转/后退/右转 |
| Q/E | 小车逆时针/顺时针旋转 |
| ↑↓←→ 或 8/2/4/6 | 摄像头方向控制 |

## 树莓派部署

### 环境要求

- Raspberry Pi 4 Model B（Cortex-A72）
- Debian 13 (trixie)
- Python 3.13 + venv
- USB 摄像头（支持 MJPEG 格式）
- 官方 5V/3A 电源 + 短粗 USB-C 线

### 部署步骤

```bash
# 1. 创建虚拟环境
cd ~/YOLO_Learn
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 2. 安装依赖
pip install onnxruntime opencv-python numpy pyserial
pip install flask flask-socketio

# 3. 设置 Web 服务开机自启
sudo cp web_server/yolo-web.service /etc/systemd/system/
sudo systemctl enable --now yolo-web.service

# 4. 设置 CPU 性能模式（提升推理稳定性）
sudo cp cpu-performance.service /etc/systemd/system/
sudo systemctl enable cpu-performance.service

# 5. 浏览网页
# http://<树莓派IP>:5000
```

### 配置文件 (`/boot/firmware/config.txt`)

```ini
enable_uart=1
max_usb_current=1
# over_voltage=2  # 如遇欠压可取消注释
```

### 串口配置

```bash
sudo sed -i 's/console=serial0,115200 //' /boot/cmdline.txt
sudo systemctl disable serial-getty@ttyS0.service
```

## 串口通信协议

### 上行（树莓派 → ESP32）

```
检测数据: F=<帧号>,P=<植物数>,B=<x1,y1,x2,y2,conf>;...
方向键:   w a s d q e x (裸发，无换行)
状态:     RUNNING / PAUSED
```

### 下行（ESP32 → 树莓派）

```
开始回传:  start
停止回传:  stop
```

方向键数据始终由 Web 透传到串口，不依赖 ESP32 的 start/stop 状态。
检测数据帧仅在 ESP32 发出 `start` 后回传，发出 `stop` 后停止回传。
Web 面板的 `[RX]` 日志实时显示所有来自 ESP32 的串口数据。

## SSH 远程连接

| 场景 | 命令 | IP |
|------|------|----|
| 手机热点 WIN | `python ssh_pi.py "命令"` | `172.20.255.185` |
| 校园网 DHU-1X | `python ssh_pi_dhu.py "命令"` | `10.206.190.162` |

已配置 SSH 密钥，免密登录。

## 性能指标

| 指标 | 数值 |
|------|------|
| 模型 | YOLOv8n → ONNX（12MB） |
| 输入尺寸 | 416×416 |
| 推理耗时 | ~232ms/帧 |
| FPS | ~4.2 |
| ONNX 线程数 | 3（留 1 核给 MJPEG） |
| 摄像头格式 | MJPEG（USB 带宽 ~2MB/s） |
| OpenMP | `OMP_WAIT_POLICY=ACTIVE` |

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 帧数忽高忽低 | 摄像头 YUYV DMA 干扰 ONNX | 设置 `CAP_PROP_FOURCC='MJPG'` |
| CPU 锁在 600MHz | 电源/USB 线供电不足 | 换官方 5V/3A + 短粗线 |
| USB 反复过流 | 摄像头功耗过高 | `max_usb_current=1` + 移除 USB Hub |
| VNC 无法显示桌面 | Wayland 不兼容 RealVNC | `/etc/lightdm/lightdm.conf` 改 `rpd-x` |
| cv2.imshow 无效 | SSH 无 DISPLAY | 设置 `DISPLAY=:0` |
| ONNX 首帧极慢 | 冷缓存 | 脚本已内置 30 帧预热 |

## 备份清单

| 文件 | 说明 |
|------|------|
| `test_camera_onnx_headless.py.bak.20250630` | 原版（285 行，无 MJPEG） |
| `test_camera_onnx_headless.py.bak.v10_final` | 当前稳定版 |

## 依赖

- Python 3.8+
- onnxruntime
- opencv-python
- numpy
- pyserial
- Flask + flask-socketio
- paramiko（SSH 工具）

## License

MIT
