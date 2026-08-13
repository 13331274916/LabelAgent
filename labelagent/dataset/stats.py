"""数据集类别统计。"""

from __future__ import annotations

from labelagent.core.models import DatasetStats
from labelagent.core.store import ProjectStore


def compute_stats(store: ProjectStore) -> DatasetStats:
    """统计当前标注数据：图像总数、标注总数、各类别标注数量。"""
    stats = DatasetStats()
    for item in store.list_images():
        stats.image_count += 1
        for ann in item.annotations:
            stats.annotation_count += 1
            stats.per_class[ann.label] = stats.per_class.get(ann.label, 0) + 1
    return stats
