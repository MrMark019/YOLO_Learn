YOLO 室内盆栽检测 - 部署包
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
