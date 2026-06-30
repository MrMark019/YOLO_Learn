#!/bin/bash
cd /home/mark/YOLO_Learn
source venv/bin/activate
# 自动探测 USB 摄像头设备号
CAM=$(v4l2-ctl --list-devices 2>/dev/null | grep -A2 'HD video' | grep -o '/dev/video[0-9]*' | head -1 | grep -o '[0-9]*')
if [ -z "$CAM" ]; then
    CAM=0
fi
echo "Using camera /dev/video$CAM"
DISPLAY=:0 python3 test_camera_onnx.py $CAM
