"""
一键打包脚本：将项目精简为树莓派/边缘设备部署包
"""

import os
import shutil
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "YOLO_Pi_Deploy"
ZIP_PATH = REPO_ROOT / "YOLO_Pi_Deploy.zip"

# 必须包含的文件和目录
INCLUDE = [
    "runs/detect/indoor_potted_plant_pi/weights/best.pt",
    "test_indoor_plant.py",
    "test_camera_ncnn.py",
    "benchmark_ncnn.py",
    "testing_pic/image.png",
    "testing_pic/屏幕截图 2026-04-15 181022.png",
    "README_Indoor_Plant.md",
]

# 部署时会生成的文件（由本脚本创建）
DEPLOY_FILES = [
    ("setup_and_run.sh", "setup_and_run.sh"),
    ("setup_and_run.bat", "setup_and_run.bat"),
    ("README.txt", "README.txt"),
]


SETUP_SH = r'''#!/bin/bash
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
'''


SETUP_BAT = r'''@echo off
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
'''


README_TXT = '''YOLO 室内盆栽检测 - 部署包
==================================

文件说明
--------
best.pt              训练好的 YOLOv8n 模型（室内盆栽检测）
test_indoor_plant.py PyTorch 摄像头/图片检测脚本
test_camera_ncnn.py  NCNN 摄像头检测脚本（当前有 bug，暂不推荐）
benchmark_ncnn.py    速度对比测试脚本
testing_pic/         2 张测试图片

使用方式
--------

【树莓派 / Linux】
1. 打开终端，进入本目录
2. 运行: bash setup_and_run.sh
3. 脚本会自动安装依赖、验证模型、然后询问是否启动摄像头

【Windows】
1. 双击运行 setup_and_run.bat
2. 脚本会自动安装依赖、验证模型、然后询问是否启动摄像头

手动运行
--------
# 单张图片测试
python test_indoor_plant.py predict testing_pic/image.png

# 摄像头实时检测
python test_indoor_plant.py camera 0

注意事项
--------
- 树莓派上预期 FPS: 3~5 (PyTorch CPU)
- 模型检测类别: potted plant（盆栽整体）
- 摄像头检测按 'q' 或 ESC 退出，按 's' 手动保存当前帧
'''


def main():
    print("=" * 60)
    print("YOLO 室内盆栽检测 - 一键打包")
    print("=" * 60)

    # 清理旧目录
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"清理旧目录: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 复制项目文件
    for rel_path in INCLUDE:
        src = REPO_ROOT / rel_path
        dst = OUTPUT_DIR / rel_path
        if not src.exists():
            print(f"警告：源文件不存在，跳过: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"复制: {rel_path}")

    # 写入部署脚本
    for filename, _ in DEPLOY_FILES:
        dst = OUTPUT_DIR / filename
        if filename == "setup_and_run.sh":
            content = SETUP_SH
        elif filename == "setup_and_run.bat":
            content = SETUP_BAT
        else:
            content = README_TXT
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"创建: {filename}")

    # Linux 脚本加执行权限
    sh_path = OUTPUT_DIR / "setup_and_run.sh"
    if os.name != "nt":
        os.chmod(sh_path, 0o755)

    # 计算总大小
    total_size = sum(
        f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()
    )
    print(f"\n打包目录大小: {total_size / 1024 / 1024:.1f} MB")

    # 压缩为 zip
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in OUTPUT_DIR.rglob("*"):
            if file_path.is_file():
                arcname = str(file_path.relative_to(OUTPUT_DIR))
                zf.write(file_path, arcname)

    zip_size = ZIP_PATH.stat().st_size
    print(f"压缩包大小:   {zip_size / 1024 / 1024:.1f} MB")
    print(f"压缩包路径:   {ZIP_PATH}")
    print("=" * 60)
    print("打包完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
