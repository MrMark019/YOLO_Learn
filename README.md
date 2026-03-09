# Raspberry Pi OpenCV 方框（口）识别

这个项目提供一个可直接在树莓派上运行的 OpenCV 脚本，用来识别画面中的方框形（`口`）。识别逻辑会同时寻找：

- 一个近似正方形/矩形的外轮廓
- 一个位于其中的内轮廓
- 内外轮廓形成“空心方框”

## 文件

- `square_frame_detector.py`：主程序

## 树莓派安装依赖

建议先用系统包安装 OpenCV：

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy
```

如果你的树莓派还没有启用摄像头，先执行：

```bash
sudo raspi-config
```

然后在界面中启用 Camera，重启系统。

## 运行方式

实时摄像头识别：

```bash
python3 square_frame_detector.py
```

如果默认摄像头不是 `0`：

```bash
python3 square_frame_detector.py --camera 1
```

如果想在测试图上验证：

```bash
python3 square_frame_detector.py --image test.png
```

## 可调参数

- `--min-area`：最小外轮廓面积
- `--aspect-tolerance`：对正方形比例的容差
- `--approx-epsilon`：轮廓近似参数
- `--threshold-block-size`：自适应阈值块大小
- `--threshold-c`：自适应阈值常量

例如：

```bash
python3 square_frame_detector.py --min-area 4000 --aspect-tolerance 0.45
```

## 退出

运行时按 `q` 退出。
