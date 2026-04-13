"""
YOLO 植物器官检测训练脚本
数据集：Nature3 - Leaf, Flower, and Fruit Detection
类别：Leaf (叶子), flower (花朵), fruit (果实)
"""

import os
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.autobatch import check_train_batch_size


REPO_ROOT = Path(__file__).resolve().parent
DATA_YAML = REPO_ROOT / "data.yaml"
RUNS_DIR = REPO_ROOT / "runs" / "detect"
DEFAULT_SINGLE_GPU_MODEL = "yolov8n.pt"
DEFAULT_DUAL_H100_MODEL = "yolov8l.pt"
DEFAULT_RUN_NAME = os.getenv("YOLO_RUN_NAME", "plant_organ_detection_h100x2")


def resolve_dataset_root() -> Path:
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_config["path"])
    if not dataset_root.is_absolute():
        dataset_root = REPO_ROOT / dataset_root
    return dataset_root.resolve()


def count_training_images(dataset_root: Path) -> int:
    image_dir = dataset_root / "train" / "images"
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    return sum(len(list(image_dir.glob(pattern))) for pattern in patterns)


def dataset_size_bytes(dataset_root: Path) -> int:
    return sum(path.stat().st_size for path in dataset_root.rglob("*") if path.is_file())


def choose_devices() -> str | int:
    override = os.getenv("YOLO_DEVICES")
    if override:
        return override

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return "0,1"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def world_size_from_devices(devices: str | int) -> int:
    if isinstance(devices, str):
        if devices in {"cpu", "mps"}:
            return 0
        return len([item for item in devices.split(",") if item.strip()])
    if isinstance(devices, int):
        return 1
    return 0


def choose_model_weights(world_size: int) -> str:
    override = os.getenv("YOLO_MODEL")
    if override:
        return override

    if world_size >= 2:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(min(2, torch.cuda.device_count()))]
        if all("H100" in name for name in gpu_names):
            return DEFAULT_DUAL_H100_MODEL
    return DEFAULT_SINGLE_GPU_MODEL


def cache_mode(dataset_root: Path) -> str | bool:
    override = os.getenv("YOLO_CACHE")
    if override:
        lowered = override.strip().lower()
        if lowered in {"false", "0", "none", "off"}:
            return False
        return override

    required_bytes = dataset_size_bytes(dataset_root) * 3
    available_bytes = psutil.virtual_memory().available
    return "ram" if available_bytes > required_bytes else False


def autotune_global_batch(model_weights: str, imgsz: int, world_size: int, dataset_size: int) -> int:
    override = os.getenv("YOLO_BATCH")
    if override:
        return int(override)

    if not torch.cuda.is_available():
        return 16

    probe_device = "cuda:0"
    target_fraction = 0.70 if world_size >= 2 else 0.60
    probe_model = YOLO(model_weights)
    probe_model.model.to(probe_device)
    try:
        per_gpu_batch = check_train_batch_size(
            probe_model.model,
            imgsz=imgsz,
            amp=True,
            batch=target_fraction,
            dataset_size=dataset_size,
        )
    finally:
        probe_model.model.to("cpu")
        del probe_model
        torch.cuda.empty_cache()

    per_gpu_batch = max(per_gpu_batch, 8)
    return per_gpu_batch * max(world_size, 1)


def build_training_profile() -> tuple[str, dict[str, Any], dict[str, Any]]:
    dataset_root = resolve_dataset_root()
    devices = choose_devices()
    world_size = world_size_from_devices(devices)
    imgsz = int(os.getenv("YOLO_IMGSZ", "640"))
    epochs = int(os.getenv("YOLO_EPOCHS", "100"))
    model_weights = choose_model_weights(world_size)
    total_train_images = count_training_images(dataset_root)
    batch = autotune_global_batch(model_weights, imgsz, world_size, total_train_images)
    workers = int(os.getenv("YOLO_WORKERS", "8" if world_size >= 2 else "4"))
    run_name = DEFAULT_RUN_NAME

    training_config = {
        "data": str(DATA_YAML),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": devices,
        "workers": workers,
        "optimizer": "SGD",
        "lr0": float(os.getenv("YOLO_LR0", "0.01")),
        "patience": 50,
        "save": True,
        "project": str(RUNS_DIR),
        "name": run_name,
        "exist_ok": False,
        "verbose": True,
        "pretrained": True,
        "cache": cache_mode(dataset_root),
        "deterministic": False if world_size >= 2 else True,
        "nbs": batch,
    }

    profile = {
        "dataset_root": str(dataset_root),
        "devices": devices,
        "world_size": world_size,
        "model_weights": model_weights,
        "train_images": total_train_images,
        "batch_per_gpu": batch // max(world_size, 1),
        "run_name": run_name,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(world_size)] if world_size else [],
    }
    return model_weights, training_config, profile


def train_yolo():
    """训练 YOLO 模型"""

    print("=" * 60)
    print("开始 YOLO 植物器官检测训练")
    print("=" * 60)

    model_weights, training_config, profile = build_training_profile()

    # 1. 加载预训练模型
    print(f"\n[1/4] 加载预训练模型 {model_weights}...")
    model = YOLO(model_weights)
    print("✓ 模型加载完成")

    # 2. 训练配置
    print("\n[2/4] 配置训练参数...")
    print("\n硬件优化配置:")
    print(f"  设备: {profile['devices']}")
    if profile["gpu_names"]:
        print(f"  GPU: {', '.join(profile['gpu_names'])}")
    print(f"  world_size: {profile['world_size']}")
    print(f"  数据集目录: {profile['dataset_root']}")
    print(f"  训练图像数: {profile['train_images']}")
    print(f"  模型: {profile['model_weights']}")
    print(f"  每卡 batch: {profile['batch_per_gpu']}")
    # 打印配置信息
    print("\n训练配置:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")

    # 3. 开始训练
    print("\n[3/4] 开始训练...")
    print("当前已启用双卡/大显存优化，训练速度会明显高于原始配置。")
    print("\n监控指标:")
    print("  - mAP@50: 目标 > 0.85")
    print("  - mAP@50-95: 目标 > 0.60")
    print("  - loss: 观察是否收敛\n")

    results = model.train(**training_config)

    # 4. 训练完成
    print("\n[4/4] 训练完成！")
    print("=" * 60)
    print("训练结果已保存到:")
    print(f"  runs/detect/{profile['run_name']}/")
    print("\n重要文件:")
    print("  - weights/best.pt  ← 最佳模型权重")
    print("  - weights/last.pt  ← 最后一次权重")
    print("  - results.png      ← 训练曲线")
    print("  - confusion_matrix.png ← 混淆矩阵")
    print("=" * 60)

    return results


def validate_model():
    """验证训练好的模型"""
    print("\n开始模型验证...")

    # 加载最佳模型
    model_path = RUNS_DIR / DEFAULT_RUN_NAME / "weights" / "best.pt"
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return

    model = YOLO(model_path)

    # 在验证集上评估
    metrics = model.val()

    print("\n验证结果:")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")

    return metrics


if __name__ == '__main__':
    # 开始训练
    train_results = train_yolo()

    # 可选：验证模型
    # validate_model()
