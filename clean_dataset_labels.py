"""
自动清洗 YOLO 标注：
1. 保留合法的 5 列检测框标注
2. 将合法类别的多边形分割标注转成检测框
3. 删除非法类别、非法坐标和损坏行
4. 备份所有被修改的原始标注文件
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


EPSILON = 1e-6


def load_dataset_config(repo_root: Path) -> tuple[Path, int]:
    config_path = repo_root / "data.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    return dataset_root.resolve(), int(config["nc"])


def clamp_unit_interval(value: float) -> float | None:
    if -EPSILON <= value <= 1 + EPSILON:
        return min(1.0, max(0.0, value))
    return None


def format_box_line(class_id: int, x_center: float, y_center: float, width: float, height: float) -> str:
    return f"{class_id} {x_center:.10f} {y_center:.10f} {width:.10f} {height:.10f}"


def clean_box_line(parts: list[str], class_count: int, original_line: str) -> tuple[str | None, str | None]:
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None, "non_numeric"

    class_value = values[0]
    if not class_value.is_integer():
        return None, "class_not_integer"

    class_id = int(class_value)
    if not 0 <= class_id < class_count:
        return None, "invalid_class"

    coords = [clamp_unit_interval(value) for value in values[1:]]
    if any(value is None for value in coords):
        return None, "coord_out_of_range"

    x_center, y_center, width, height = coords
    if width <= 0 or height <= 0:
        return None, "non_positive_box"

    return original_line, None


def clean_polygon_line(parts: list[str], class_count: int) -> tuple[str | None, str | None]:
    if len(parts) <= 5:
        return None, "malformed_polygon"
    if len(parts) % 2 == 0:
        return None, "malformed_polygon"

    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None, "non_numeric"

    class_value = values[0]
    if not class_value.is_integer():
        return None, "class_not_integer"

    class_id = int(class_value)
    if not 0 <= class_id < class_count:
        return None, "invalid_class"

    raw_coords = values[1:]
    if len(raw_coords) < 6:
        return None, "too_few_polygon_points"

    coords = [clamp_unit_interval(value) for value in raw_coords]
    if any(value is None for value in coords):
        return None, "coord_out_of_range"

    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None, "non_positive_box"

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return format_box_line(class_id, x_center, y_center, width, height), None


def clean_label_file(label_path: Path, class_count: int) -> tuple[str, dict[str, int], list[dict[str, str | int]]]:
    original_text = label_path.read_text(encoding="utf-8")
    cleaned_lines: list[str] = []
    file_stats: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []

    for line_number, raw_line in enumerate(original_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            file_stats["blank_lines_removed"] += 1
            continue

        parts = stripped.split()
        cleaned_line = None
        reason = None

        if len(parts) == 5:
            cleaned_line, reason = clean_box_line(parts, class_count, stripped)
            if cleaned_line is not None:
                file_stats["valid_boxes_kept"] += 1
        elif len(parts) > 5:
            cleaned_line, reason = clean_polygon_line(parts, class_count)
            if cleaned_line is not None:
                file_stats["polygons_converted_to_boxes"] += 1
        else:
            reason = "too_few_fields"

        if cleaned_line is not None:
            cleaned_lines.append(cleaned_line)
            continue

        file_stats[f"dropped_{reason}"] += 1
        if len(examples) < 5:
            examples.append(
                {
                    "line_number": line_number,
                    "reason": reason,
                    "content": stripped[:240],
                }
            )

    cleaned_text = "\n".join(cleaned_lines)
    if cleaned_lines:
        cleaned_text += "\n"

    return cleaned_text, dict(file_stats), examples


def remove_cache_files(dataset_root: Path) -> list[str]:
    removed = []
    for cache_path in sorted(dataset_root.rglob("*.cache")):
        cache_path.unlink(missing_ok=True)
        removed.append(str(cache_path))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="自动清洗 YOLO 数据集标注")
    parser.add_argument("--dry-run", action="store_true", help="只扫描，不落盘修改")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    dataset_root, class_count = load_dataset_config(repo_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    backup_root = dataset_root / "cleaning_backup" / timestamp
    report_root = repo_root / "cleaning_reports"
    report_root.mkdir(exist_ok=True)
    report_path = report_root / f"label_clean_report_{timestamp}.json"

    totals: Counter[str] = Counter()
    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    report_examples: dict[str, list[dict[str, str | int]]] = {}
    modified_files: list[str] = []
    emptied_files: list[str] = []

    print("=" * 60)
    print("开始清洗数据集标注")
    print("=" * 60)
    print(f"数据集目录: {dataset_root}")
    print(f"类别数量: {class_count}")
    print(f"模式: {'dry-run' if args.dry_run else 'apply'}")

    for split in ("train", "valid", "test"):
        labels_dir = dataset_root / split / "labels"
        label_files = sorted(labels_dir.glob("*.txt"))
        print(f"\n[{split}] 扫描 {len(label_files)} 个标注文件...")

        for label_path in label_files:
            totals["files_scanned"] += 1
            per_split[split]["files_scanned"] += 1

            cleaned_text, file_stats, examples = clean_label_file(label_path, class_count)
            original_text = label_path.read_text(encoding="utf-8")

            for key, value in file_stats.items():
                totals[key] += value
                per_split[split][key] += value

            if cleaned_text == original_text:
                continue

            totals["files_modified"] += 1
            per_split[split]["files_modified"] += 1
            modified_files.append(str(label_path))

            if cleaned_text == "":
                totals["files_emptied"] += 1
                per_split[split]["files_emptied"] += 1
                emptied_files.append(str(label_path))

            if examples:
                report_examples[str(label_path)] = examples

            if args.dry_run:
                continue

            backup_path = backup_root / split / "labels" / label_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, backup_path)
            label_path.write_text(cleaned_text, encoding="utf-8")

    removed_caches = [] if args.dry_run else remove_cache_files(dataset_root)

    report = {
        "timestamp_utc": timestamp,
        "dataset_root": str(dataset_root),
        "class_count": class_count,
        "dry_run": args.dry_run,
        "backup_root": None if args.dry_run else str(backup_root),
        "report_path": str(report_path),
        "totals": dict(totals),
        "per_split": {split: dict(stats) for split, stats in per_split.items()},
        "removed_cache_files": removed_caches,
        "modified_files": modified_files,
        "emptied_files": emptied_files,
        "examples": report_examples,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("清洗完成")
    print("=" * 60)
    print(f"扫描文件数: {totals['files_scanned']}")
    print(f"修改文件数: {totals['files_modified']}")
    print(f"保留检测框: {totals['valid_boxes_kept']}")
    print(f"分割转检测框: {totals['polygons_converted_to_boxes']}")
    print(f"删除非法类别: {totals['dropped_invalid_class']}")
    print(f"删除格式问题: {totals['dropped_malformed_polygon'] + totals['dropped_too_few_fields'] + totals['dropped_non_numeric'] + totals['dropped_class_not_integer']}")
    print(f"删除越界坐标: {totals['dropped_coord_out_of_range']}")
    print(f"删除非正框: {totals['dropped_non_positive_box']}")
    print(f"清空标签文件数: {totals['files_emptied']}")
    if removed_caches:
        print(f"已删除缓存文件: {len(removed_caches)}")
    if not args.dry_run:
        print(f"备份目录: {backup_root}")
    print(f"报告文件: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
