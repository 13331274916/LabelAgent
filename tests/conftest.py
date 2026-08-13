"""pytest 共享夹具与辅助函数。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保仓库根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from labelagent.core.models import Annotation, ImageItem, Point  # noqa: E402


def make_image(path: Path, size=(320, 240), color="gray", draw_rect: tuple | None = None) -> Path:
    """生成一张测试图像。"""
    img = Image.new("RGB", size, color)
    if draw_rect:
        d = ImageDraw.Draw(img)
        d.rectangle(draw_rect, fill="red")
    img.save(path, "PNG")
    return path


def make_item(
    path: str,
    width: int = 320,
    height: int = 240,
    annotations: list[Annotation] | None = None,
) -> ImageItem:
    return ImageItem(
        path=path,
        filename=Path(path).name,
        width=width,
        height=height,
        annotations=annotations or [],
    )


def make_annotation(label: str = "scratch_defect", bbox=(10, 10, 60, 60)) -> Annotation:
    x1, y1, x2, y2 = bbox
    return Annotation(
        label=label,
        points=[
            Point(x=x1, y=y1),
            Point(x=x2, y=y1),
            Point(x=x2, y=y2),
            Point(x=x1, y=y2),
        ],
    )


@pytest.fixture()
def tmp_project(tmp_path):
    """构造一个含两张标注图像的临时项目。"""
    from labelagent.core.store import ProjectStore

    p1 = make_image(tmp_path / "a.png")
    p2 = make_image(tmp_path / "b.png", color="lightblue")
    store = ProjectStore()
    store.add_images(
        [
            make_item(str(p1), annotations=[make_annotation("scratch_defect", (10, 10, 60, 60))]),
            make_item(str(p2), annotations=[make_annotation("crack", (30, 30, 90, 70))]),
        ]
    )
    return store, tmp_path
