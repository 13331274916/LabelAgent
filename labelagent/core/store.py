"""ProjectStore：LabelAgent 的内存项目状态中心。

所有模块共享同一个 ProjectStore 实例（由 API 应用创建并注入）：
- 保存已导入图像列表与标注；
- 提供线程安全的新增/查询/修改/删除操作；
- 提供导出、统计、划分所需的数据快照。
"""

from __future__ import annotations

import threading
from typing import Optional

from labelagent.core.models import Annotation, ImageItem


class ProjectStore:
    """线程安全的项目状态仓库。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._images: dict[str, ImageItem] = {}

    # ------------------------------------------------------------------
    # 图像管理
    # ------------------------------------------------------------------
    def add_image(self, item: ImageItem) -> ImageItem:
        with self._lock:
            self._images[item.id] = item
            return item

    def add_images(self, items: list[ImageItem]) -> list[ImageItem]:
        with self._lock:
            for it in items:
                self._images[it.id] = it
            return items

    def get(self, image_id: str) -> Optional[ImageItem]:
        with self._lock:
            return self._images.get(image_id)

    def list_images(self) -> list[ImageItem]:
        with self._lock:
            return list(self._images.values())

    def remove(self, image_id: str) -> Optional[ImageItem]:
        with self._lock:
            return self._images.pop(image_id, None)

    def remove_many(self, image_ids: list[str]) -> int:
        with self._lock:
            n = 0
            for i in image_ids:
                if self._images.pop(i, None) is not None:
                    n += 1
            return n

    def clear(self) -> None:
        with self._lock:
            self._images.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._images)

    # ------------------------------------------------------------------
    # 标注管理
    # ------------------------------------------------------------------
    def set_annotations(self, image_id: str, annotations: list[Annotation]) -> Optional[ImageItem]:
        """整体替换一张图的标注（Agent 标注结果写入使用）。"""
        with self._lock:
            item = self._images.get(image_id)
            if item is None:
                return None
            item.annotations = annotations
            return item

    def add_annotation(self, image_id: str, annotation: Annotation) -> Optional[Annotation]:
        with self._lock:
            item = self._images.get(image_id)
            if item is None:
                return None
            item.annotations.append(annotation)
            return annotation

    def delete_annotation(self, image_id: str, annotation_id: str) -> bool:
        with self._lock:
            item = self._images.get(image_id)
            if item is None:
                return False
            before = len(item.annotations)
            item.annotations = [a for a in item.annotations if a.id != annotation_id]
            return len(item.annotations) < before

    def update_annotation(
        self,
        image_id: str,
        annotation_id: str,
        *,
        label: Optional[str] = None,
        points: Optional[list[dict]] = None,
        confidence: Optional[float] = None,
    ) -> Optional[Annotation]:
        """更新单条标注（支持标签重命名 / 顶点修订 / 置信度修改）。"""
        with self._lock:
            item = self._images.get(image_id)
            if item is None:
                return None
            for ann in item.annotations:
                if ann.id == annotation_id:
                    upd: dict = {}
                    if label is not None:
                        upd["label"] = label
                    if points is not None:
                        from labelagent.core.models import Point

                        upd["points"] = [Point(**p) for p in points]
                    if confidence is not None:
                        upd["confidence"] = confidence
                    ann = ann.model_copy(update=upd)
                    # 替换列表中的原对象
                    idx = item.annotations.index(next(a for a in item.annotations if a.id == annotation_id))
                    item.annotations[idx] = ann
                    return ann
            return None

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, ImageItem]:
        """返回所有图像（用于批量导出/统计）。"""
        with self._lock:
            return dict(self._images)

    def class_list(self) -> list[str]:
        """当前项目中出现过的全部类别（保持出现顺序）。"""
        seen: list[str] = []
        with self._lock:
            for item in self._images.values():
                for ann in item.annotations:
                    if ann.label and ann.label not in seen:
                        seen.append(ann.label)
        return seen
