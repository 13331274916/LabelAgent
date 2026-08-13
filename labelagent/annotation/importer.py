"""图像导入：支持整个文件夹或一次多张图片。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

from labelagent.config import IMAGE_EXTS
from labelagent.core.models import ImageItem


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def _read_image_size(path: str) -> tuple[int, int]:
    """读取图像尺寸；失败时返回 (0, 0)。"""
    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except (UnidentifiedImageError, OSError, FileNotFoundError):
        return (0, 0)


def scan_folder(folder: str, recursive: bool = True) -> list[str]:
    """扫描文件夹下的全部图像文件路径（保持稳定排序）。"""
    root = Path(folder)
    if not root.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    files = [p for p in root.glob(pattern) if p.is_file() and is_image_file(str(p))]
    return [str(p) for p in sorted(files)]


def import_paths(paths: list[str], store) -> list[ImageItem]:
    """将文件路径列表导入 ProjectStore（已存在的路径跳过）。"""
    items: list[ImageItem] = []
    existing = {i.path for i in store.list_images()}
    for p in paths:
        if p in existing:
            continue
        if not Path(p).is_file():
            continue
        width, height = _read_image_size(p)
        if width <= 0 or height <= 0:
            continue
        item = ImageItem(
            path=p,
            filename=Path(p).name,
            width=width,
            height=height,
        )
        items.append(item)
    store.add_images(items)
    return items


def import_folder(folder: str, store, recursive: bool = True) -> list[ImageItem]:
    """导入整个文件夹的图像。"""
    files = scan_folder(folder, recursive=recursive)
    return import_paths(files, store)


def import_single_file(path: str, store) -> Optional[ImageItem]:
    """导入单张图片。"""
    items = import_paths([path], store)
    return items[0] if items else None
