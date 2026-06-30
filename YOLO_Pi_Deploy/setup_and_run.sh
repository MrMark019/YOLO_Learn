#!/bin/bash
set -e

echo "=================================="
echo "YOLO 室内盆栽检测 - 一键部署"
echo "=================================="

PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "错误：未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

echo "Python 版本:"
$PYTHON_CMD --version

echo ""
echo "正在安装依赖（可能需要几分钟）..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install ultralytics opencv-python || {
    echo "pip 安装失败，尝试使用 apt 安装 OpenCV..."
    sudo apt-get update
    sudo apt-get install -y python3-opencv
    $PYTHON_CMD -m pip install ultralytics
}

echo ""
echo "验证模型加载..."
$PYTHON_CMD -c "
from ultralytics import YOLO
import os
model_path = 'runs/detect/indoor_potted_plant_pi/weights/best.pt'
if not os.path.exists(model_path):
    print(f'错误：模型文件不存在：{model_path}')
    exit(1)
model = YOLO(model_path)
print('模型加载成功！')
results = model('testing_pic/image.png', imgsz=320, verbose=False)
print(f'测试图检测到 {len(results[0].boxes)} 个盆栽')
"

echo ""
echo "=================================="
echo "部署完成！"
echo "=================================="
echo "可用命令："
echo "  $PYTHON_CMD test_indoor_plant.py camera 0     # 启动摄像头检测"
echo "  $PYTHON_CMD test_indoor_plant.py predict <图>  # 单张图片检测"
echo ""
read -p "是否立即启动摄像头检测？(y/n): " answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    $PYTHON_CMD test_indoor_plant.py camera 0
fi
