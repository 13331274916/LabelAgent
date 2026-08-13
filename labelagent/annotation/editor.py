"""标注修订：标签重命名、删除错误标注、更新顶点等基础操作。"""

from __future__ import annotations

from typing import Optional

from labelagent.core.models import Annotation
from labelagent.core.store import ProjectStore


def rename_label(store: ProjectStore, image_id: str, annotation_id: str, new_label: str) -> Optional[Annotation]:
    """修改已有标注的标签名称。"""
    return store.update_annotation(image_id, annotation_id, label=new_label)


def delete_annotation(store: ProjectStore, image_id: str, annotation_id: str) -> bool:
    """删除错误标注。"""
    return store.delete_annotation(image_id, annotation_id)


def update_points(store: ProjectStore, image_id: str, annotation_id: str, points: list[dict]) -> Optional[Annotation]:
    """更新标注顶点（预留手工微调接口）。"""
    return store.update_annotation(image_id, annotation_id, points=points)


def add_annotation(store: ProjectStore, image_id: str, label: str, points: list[dict]) -> Optional[Annotation]:
    """手工新增一条标注（预留手工绘制接口）。"""
    from labelagent.core.models import Point

    ann = Annotation(label=label, points=[Point(**p) for p in points], source="manual")
    return store.add_annotation(image_id, ann)
