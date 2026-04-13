"""
数据集验证脚本
检查数据集完整性和格式正确性
"""

import os
import yaml
from pathlib import Path
from collections import Counter
from PIL import Image

def validate_dataset():
    """验证数据集结构和格式"""
    
    print("=" * 60)
    print("数据集验证")
    print("=" * 60)
    
    repo_root = Path(__file__).resolve().parent
    data_yaml_repo_path = repo_root / "data.yaml"

    with open(data_yaml_repo_path, 'r', encoding='utf-8') as f:
        repo_data_config = yaml.safe_load(f)

    dataset_root = Path(repo_data_config["path"])
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root
    
    # 1. 检查 data.yaml
    print("\n[1/6] 检查 data.yaml 配置文件...")
    data_yaml_path = dataset_root / "data.yaml"
    
    if not data_yaml_path.exists():
        print(f"❌ 错误：data.yaml 不存在：{data_yaml_path}")
        return False
    
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print("[OK] data.yaml 存在")
    print(f"  训练集路径：{data_config.get('train', 'N/A')}")
    print(f"  验证集路径：{data_config.get('val', 'N/A')}")
    print(f"  类别数：{data_config.get('nc', 'N/A')}")
    print(f"  类别名称：{data_config.get('names', 'N/A')}")
    
    # 2. 检查目录结构
    print("\n[2/6] 检查目录结构...")
    
    required_dirs = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "valid" / "images",
        dataset_root / "valid" / "labels",
        dataset_root / "test" / "images",
        dataset_root / "test" / "labels",
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"❌ 错误：目录不存在：{dir_path}")
            return False
    
    print("[OK] 所有必需目录存在")
    
    # 3. 统计文件数量
    print("\n[3/6] 统计文件数量...")
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))
        
        print(f"\n{split.upper()}:")
        print(f"  图像数量：{len(image_files)}")
        print(f"  标注数量：{len(label_files)}")
        
        if len(image_files) != len(label_files):
            print(f"  ⚠️ 警告：图像和标注数量不匹配！")
    
    # 4. 检查图像和标注文件对应关系
    print("\n[4/6] 检查图像和标注文件对应关系...")
    
    train_images = set(f.stem for f in (dataset_root / "train" / "images").glob("*.jpg"))
    train_labels = set(f.stem for f in (dataset_root / "train" / "labels").glob("*.txt"))
    
    missing_labels = train_images - train_labels
    missing_images = train_labels - train_images
    
    if missing_labels:
        print(f"⚠️ 警告：{len(missing_labels)} 个图像缺少标注")
    if missing_images:
        print(f"⚠️ 警告：{len(missing_images)} 个标注缺少图像")
    
    if not missing_labels and not missing_images:
        print("[OK] 图像和标注文件一一对应")
    
    # 5. 检查标注格式
    print("\n[5/6] 检查标注格式...")

    sample_label_file = list((dataset_root / "train" / "labels").glob("*.txt"))[0]

    with open(sample_label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  样本文件：{sample_label_file.name}")
    print(f"  标注数量：{len(lines)}")

    # 检查第一行格式
    if len(lines) > 0:
        parts = lines[0].strip().split()
        if len(parts) == 5:
            class_id, x_center, y_center, width, height = map(float, parts)
            print(f"  格式检查:")
            print(f"    类别 ID: {int(class_id)} (有效范围：0-2)")
            print(f"    X 中心：{x_center:.4f} (有效范围：0-1)")
            print(f"    Y 中心：{y_center:.4f} (有效范围：0-1)")
            print(f"    宽度：{width:.4f} (有效范围：0-1)")
            print(f"    高度：{height:.4f} (有效范围：0-1)")

            if 0 <= class_id <= 2 and all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                print("[OK] 样本标注格式正确")
            else:
                print("❌ 错误：样本标注值超出有效范围")
                return False
        else:
            print("❌ 错误：样本标注格式不正确（应该是 5 个值）")
            return False

    invalid_label_files = []
    class_counter = Counter()

    for split in splits:
        for label_file in (dataset_root / split / "labels").glob("*.txt"):
            file_has_error = False
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue

                    parts = stripped.split()
                    if len(parts) != 5:
                        invalid_label_files.append((split, label_file.name, line_number, "字段数量不是 5"))
                        file_has_error = True
                        continue

                    try:
                        class_id, x_center, y_center, width, height = map(float, parts)
                    except ValueError:
                        invalid_label_files.append((split, label_file.name, line_number, "存在非数字字段"))
                        file_has_error = True
                        continue

                    class_counter[int(class_id)] += 1
                    if int(class_id) < 0 or int(class_id) > 2:
                        invalid_label_files.append(
                            (split, label_file.name, line_number, f"类别 ID {int(class_id)} 超出 0-2")
                        )
                        file_has_error = True
                        continue

                    if not all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                        invalid_label_files.append(
                            (split, label_file.name, line_number, "坐标值超出 0-1")
                        )
                        file_has_error = True

            if file_has_error:
                continue

    print(f"  类别分布：{dict(sorted(class_counter.items()))}")
    if invalid_label_files:
        print(f"  ⚠️ 检测到 {len(invalid_label_files)} 条异常标注，以下为前 10 条：")
        for split, file_name, line_number, reason in invalid_label_files[:10]:
            print(f"    - {split}/{file_name}:{line_number} -> {reason}")
        print("  ⚠️ Ultralytics 训练时会自动跳过这些异常标注对应的样本")
    else:
        print("[OK] 所有标注均在有效范围内")
    
    # 6. 检查图像尺寸
    print("\n[6/6] 检查图像尺寸...")
    
    sample_image = list((dataset_root / "train" / "images").glob("*.jpg"))[0]
    img = Image.open(sample_image)
    width, height = img.size
    
    print(f"  样本图像：{sample_image.name}")
    print(f"  尺寸：{width} x {height}")
    print("[OK] 图像可以正常打开")
    
    # 总结
    print("\n" + "=" * 60)
    print("数据集验证完成！")
    print("=" * 60)
    print("[OK] 数据集结构正确")
    if invalid_label_files:
        print(f"[WARN] 检测到 {len(invalid_label_files)} 条异常标注")
        print("[WARN] 训练时会跳过对应异常样本")
    else:
        print("[OK] 文件格式正确")
    print("[OK] 可以开始训练")
    
    return True

if __name__ == '__main__':
    try:
        success = validate_dataset()
        if success:
            print("\n[OK] 数据集验证通过，可以运行训练脚本！")
        else:
            print("\n[ERROR] 数据集验证失败，请检查错误信息")
    except Exception as e:
        print(f"\n[ERROR] 验证过程出错：{e}")
        import traceback
        traceback.print_exc()
