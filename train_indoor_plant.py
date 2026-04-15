"""
室内盆栽植物检测 - 训练脚本
数据集：从 HomeObjects-3K 提取的 potted plant 类别
目标：树莓派轻量部署
"""

import os
from pathlib import Path

import torch
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent
DATA_YAML = REPO_ROOT / "datasets" / "indoor_potted_plant" / "data.yaml"
RUNS_DIR = REPO_ROOT / "runs" / "detect"
MODEL_WEIGHTS = "yolov8n.pt"  # 树莓派部署必须用 nano 版
RUN_NAME = "indoor_potted_plant_pi"


def train():
    print("=" * 60)
    print("室内盆栽植物检测 - 开始训练")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    print(f"数据集: {DATA_YAML}")
    print(f"预训练模型: {MODEL_WEIGHTS}")
    print(f"输出名称: {RUN_NAME}")

    # 加载模型
    model = YOLO(MODEL_WEIGHTS)

    # 训练配置（树莓派友好参数）
    training_config = {
        "data": str(DATA_YAML),
        "epochs": 100,
        "batch": 32 if torch.cuda.is_available() else 8,
        "imgsz": 416,          # 416 比 640 快很多，精度损失小
        "device": device,
        "workers": 4,
        "optimizer": "SGD",
        "lr0": 0.01,
        "patience": 30,        # 30 轮不提升则早停
        "save": True,
        "project": str(RUNS_DIR),
        "name": RUN_NAME,
        "exist_ok": True,
        "verbose": True,
        "pretrained": True,
    }

    print("\n训练参数:")
    for k, v in training_config.items():
        print(f"  {k}: {v}")

    # 开始训练
    print("\n开始训练...")
    results = model.train(**training_config)

    # 训练完成
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存路径: {RUNS_DIR / RUN_NAME / 'weights' / 'best.pt'}")
    print("=" * 60)

    # 在测试集上验证
    print("\n在测试集上进行最终验证...")
    metrics = model.val(split="test")
    print(f"\n测试结果:")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")

    return results, metrics


if __name__ == "__main__":
    train()
