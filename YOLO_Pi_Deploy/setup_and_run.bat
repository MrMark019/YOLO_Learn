@echo off
chcp 65001 >nul
echo ==================================
echo YOLO 室内盆栽检测 - 一键部署
echo ==================================

python --version 2>nul || (
    echo 错误：未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo.
echo 正在安装依赖...
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python

echo.
echo 验证模型加载...
python -c "from ultralytics import YOLO; import os; model_path='runs/detect/indoor_potted_plant_pi/weights/best.pt'; model=YOLO(model_path); print('模型加载成功！'); results=model('testing_pic/image.png', imgsz=320, verbose=False); print(f'测试图检测到 {len(results[0].boxes)} 个盆栽')"

echo.
echo ==================================
echo 部署完成！
echo ==================================
echo 可用命令：
echo   python test_indoor_plant.py camera 0      ^(启动摄像头检测^)
echo   python test_indoor_plant.py predict ^<图^>  ^(单张图片检测^)
echo.
set /p answer="是否立即启动摄像头检测？(y/n): "
if "%answer%"=="y" (
    python test_indoor_plant.py camera 0
)
