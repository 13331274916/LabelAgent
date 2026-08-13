"""Train / Val 数据集自动划分 + dataset.yaml 生成。

划分流程：
1. 统计类别并生成类别映射；
2. 按训练集比例（0.50 ~ 0.95，步长 0.05）随机划分图像；
3. 在输出目录生成 YOLO 风格数据集：
     dataset.yaml
     images/train/  images/val/   （复制图像文件）
     labels/train/  labels/val/   （YOLO TXT 标注）
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

import yaml

from labelagent.config import (
    DEFAULT_DATASET_NAME,
    SPLIT_RATIO_MAX,
    SPLIT_RATIO_MIN,
    SPLIT_RATIO_STEP,
    SPLIT_SEED,
    WORKSPACE_DIR,
)
from labelagent.annotation.export import _class_map_of, to_yolo_txt
from labelagent.core.models import ImageItem, SplitResult
from labelagent.core.store import ProjectStore


def _validate_ratio(ratio: float) -> float:
    if not (SPLIT_RATIO_MIN <= ratio <= SPLIT_RATIO_MAX):
        raise ValueError(f"训练集比例必须在 {SPLIT_RATIO_MIN} ~ {SPLIT_RATIO_MAX} 之间")
    # 按界面步长 0.05 取整
    ratio = round(ratio / SPLIT_RATIO_STEP) * SPLIT_RATIO_STEP
    return max(SPLIT_RATIO_MIN, min(SPLIT_RATIO_MAX, ratio))


def split_dataset(
    store: ProjectStore,
    train_ratio: float = 0.80,
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: Optional[str | Path] = None,
    seed: int = SPLIT_SEED,
) -> SplitResult:
    """划分数据集并生成 dataset.yaml。"""
    train_ratio = _validate_ratio(train_ratio)
    images = store.list_images()
    classes = store.class_list()

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_ratio)) if shuffled else 0
    train_items = shuffled[:n_train]
    val_items = shuffled[n_train:]

    out_dir = Path(output_dir) if output_dir else WORKSPACE_DIR / "datasets" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = _class_map_of(images)
    yaml_path = _write_dataset(
        out_dir=out_dir,
        dataset_name=dataset_name,
        train_items=train_items,
        val_items=val_items,
        class_map=mapping,
    )

    return SplitResult(
        dataset_name=dataset_name,
        train_count=len(train_items),
        val_count=len(val_items),
        train_ratio=train_ratio,
        yaml_path=str(yaml_path),
        output_dir=str(out_dir),
        classes=list(mapping.keys()),
    )


def _write_dataset(
    out_dir: Path,
    dataset_name: str,
    train_items: list[ImageItem],
    val_items: list[ImageItem],
    class_map: dict[str, int],
) -> Path:
    """复制图像并写出 YOLO 标注与 dataset.yaml。"""
    for split_name, items in (("train", train_items), ("val", val_items)):
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            try:
                shutil.copy2(item.path, img_dir / item.filename)
            except OSError:
                continue
            txt = to_yolo_txt(item, class_map)
            stem = Path(item.filename).stem
            (lbl_dir / f"{stem}.txt").write_text(txt, encoding="utf-8")

    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for name, i in sorted(class_map.items(), key=lambda kv: kv[1])},
        "nc": len(class_map),
        "dataset_name": dataset_name,
    }
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path
