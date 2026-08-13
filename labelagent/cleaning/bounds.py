"""越界标注检查与纠偏。

规则：标注 bbox 允许有少量容差（默认 2 像素或相对尺寸 1%），
超过容差记为越界（has_oob）；纠偏时把顶点裁剪回图像范围内。
"""

from __future__ import annotations

from labelagent.core.models import Annotation, ImageItem
from labelagent.core.store import ProjectStore

DEFAULT_TOLERANCE_PX = 2.0
DEFAULT_TOLERANCE_RATIO = 0.01


def annotation_oob_amount(
    ann: Annotation,
    width: int,
    height: int,
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
    tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO,
) -> float:
    """返回单条标注的越界量（像素），0 表示不越界。"""
    if not width or not height:
        return 0.0
    tol = max(tolerance_px, tolerance_ratio * max(width, height))
    worst = 0.0
    for p in ann.points:
        worst = max(worst, -p.x, -p.y, p.x - width, p.y - height)
    return max(0.0, worst - tol)


def check_out_of_bounds(
    store: ProjectStore,
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
    tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO,
) -> dict[str, int]:
    """检查全部图像的越界标注，更新 has_oob 标记。

    返回 {image_id: 越界标注数量}（仅包含有越界的图像）。
    """
    result: dict[str, int] = {}
    for item in store.list_images():
        cnt = sum(
            1
            for ann in item.annotations
            if annotation_oob_amount(ann, item.width, item.height, tolerance_px, tolerance_ratio) > 0
        )
        item.has_oob = cnt > 0
        if cnt:
            result[item.id] = cnt
    return result


def fix_out_of_bounds(
    store: ProjectStore,
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
    tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO,
) -> list[str]:
    """自动纠偏越界框：将顶点裁剪回图像范围内，返回修复的图像 id 列表。"""
    fixed: list[str] = []
    for item in store.list_images():
        oob_anns = [
            a
            for a in item.annotations
            if annotation_oob_amount(a, item.width, item.height, tolerance_px, tolerance_ratio) > 0
        ]
        if not oob_anns:
            continue
        item.annotations = [a.clamp(item.width, item.height) for a in item.annotations]
        item.has_oob = False
        fixed.append(item.id)
    return fixed
