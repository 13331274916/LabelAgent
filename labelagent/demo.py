"""演示图像生成（合成数据）。

用于没有真实数据/API Key 时完整体验 标注 → 清洗 → 数据集 → 训练 全流程。
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from labelagent.config import DEMO_IMAGE_DIR
from labelagent.core.models import ImageItem
from labelagent.core.store import ProjectStore


def generate_demo_images(
    store: ProjectStore,
    count: int = 12,
    width: int = 640,
    height: int = 480,
    out_dir: Path = DEMO_IMAGE_DIR,
) -> list[ImageItem]:
    """生成 count 张合成图像（含形状、噪声、模糊/重复样本）并导入项目。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    items: list[ImageItem] = []

    for i in range(count):
        img = Image.new("RGB", (width, height), _random_bg(rng))
        draw = ImageDraw.Draw(img)
        # 随机绘制圆形/矩形/多边形作为“缺陷”
        for _ in range(rng.randint(2, 6)):
            x = rng.randint(0, width - 1)
            y = rng.randint(0, height - 1)
            r = rng.randint(8, max(10, width // 12))
            color = _random_color(rng)
            kind = rng.choice(["ellipse", "rectangle", "polygon"])
            if kind == "ellipse":
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
            elif kind == "rectangle":
                draw.rectangle([x - r, y - r, x + r, y + r], fill=color)
            else:
                pts = [
                    (x + r * math.cos(2 * math.pi * k / 5), y + r * math.sin(2 * math.pi * k / 5))
                    for k in range(5)
                ]
                draw.polygon(pts, fill=color)
        # 部分图像模糊（用于演示清洗）
        if i % 4 == 1:
            img = img.filter(ImageFilter.GaussianBlur(radius=6))
        # 部分图像复制（用于演示重复图检测）
        prev = out_dir / f"demo_{i - 4:03d}.png"
        if i % 5 == 4 and i > 4 and prev.exists():
            img = Image.open(prev)

        path = out_dir / f"demo_{i:03d}.png"
        img.save(path, "PNG")
        items.append(
            ImageItem(
                path=str(path),
                filename=path.name,
                width=width,
                height=height,
            )
        )

    store.add_images(items)
    return items


def _random_bg(rng: random.Random) -> tuple[int, int, int]:
    base = rng.randint(160, 235)
    return (base, base, base)


def _random_color(rng: random.Random) -> tuple[int, int, int]:
    return (rng.randint(30, 200), rng.randint(30, 200), rng.randint(30, 200))
