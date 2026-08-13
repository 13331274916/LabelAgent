"""清洗模块测试：模糊检测、重复图检测、越界纠偏、自动清洗。"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageFilter

from labelagent.core.models import Annotation, Point
from tests.conftest import make_image, make_item


def test_blurry_detection(tmp_path):
    """纯色图 Laplacian 方差接近 0，应判定为模糊；随机噪声图应清晰。"""
    import random

    from labelagent.cleaning.quality import diagnose
    from labelagent.core.store import ProjectStore

    solid = make_image(tmp_path / "solid.png", color="gray")
    # 随机噪声图像素级锐利，应判定为清晰
    rng = random.Random(7)
    img = Image.new("RGB", (320, 240))
    px = img.load()
    for y in range(240):
        for x in range(320):
            v = rng.randint(0, 255)
            px[x, y] = (v, v, v)
    noise_path = tmp_path / "noise.png"
    img.save(noise_path, "PNG")

    store = ProjectStore()
    store.add_images([make_item(str(solid)), make_item(str(noise_path))])
    result = diagnose(store, blur_threshold=100.0)
    by_name = {it.filename: it.is_blurry for it in store.list_images()}
    assert by_name["solid.png"] is True
    assert by_name["noise.png"] is False
    assert result.blurry_count == 1
    assert 0 <= result.health_score <= 100


def test_duplicate_detection(tmp_path):
    """完全相同/近似的图像应被识别为重复，不同图像不误报。"""
    from labelagent.cleaning.duplicate import HashCache, detect_duplicates
    from labelagent.core.store import ProjectStore

    # 不同内容的图像（避免纯色图 dHash 恒为 0 的边界情况）
    a = make_image(tmp_path / "a.png", draw_rect=(20, 20, 120, 100))
    b = make_image(tmp_path / "b.png", draw_rect=(20, 20, 120, 100))
    c = make_image(tmp_path / "c.png", color="lightblue", draw_rect=(200, 150, 300, 220))
    # b 是 a 的完全拷贝
    shutil.copy2(a, b)

    store = ProjectStore()
    store.add_images([make_item(str(a)), make_item(str(b)), make_item(str(c))])
    cache = HashCache(tmp_path / "cache.json")

    dups_lsh = detect_duplicates(store.list_images(), method="lsh", cache=cache)
    ids = [it.id for it in store.list_images()]
    # a 与 b 重复（其中一张是另一张的副本）
    assert len(dups_lsh) == 1

    dups_pair = detect_duplicates(store.list_images(), method="pairwise")
    assert len(dups_pair) == 1
    # c 不应出现在重复表中
    assert ids[2] not in dups_pair


def test_out_of_bounds_fix(tmp_path):
    from labelagent.cleaning.bounds import check_out_of_bounds, fix_out_of_bounds
    from labelagent.core.store import ProjectStore

    img = make_image(tmp_path / "oob.png")
    # 越界标注：顶点超过图像范围
    oob_ann = Annotation(
        label="oob",
        points=[
            Point(x=10, y=10),
            Point(x=400, y=10),
            Point(x=400, y=300),
            Point(x=10, y=300),
        ],
    )
    store = ProjectStore()
    item = make_item(str(img), annotations=[oob_ann])
    store.add_image(item)

    oob_map = check_out_of_bounds(store)
    assert item.id in oob_map
    assert item.has_oob is True

    fixed = fix_out_of_bounds(store)
    assert item.id in fixed
    assert item.has_oob is False
    assert item.annotations[0].bbox[2] <= 320
    assert item.annotations[0].bbox[3] <= 240


def test_auto_clean(tmp_path):
    from labelagent.cleaning.cleaner import auto_clean
    from labelagent.cleaning.duplicate import HashCache
    from labelagent.core.store import ProjectStore

    a = make_image(tmp_path / "a.png")
    b = make_image(tmp_path / "b.png")
    shutil.copy2(a, b)
    blur = make_image(tmp_path / "blur.png")
    img = Image.open(blur)
    img.filter(ImageFilter.GaussianBlur(12)).save(blur)
    img.close()

    store = ProjectStore()
    items = [make_item(str(a)), make_item(str(b)), make_item(str(blur))]
    store.add_images(items)
    for it in store.list_images():
        it.is_blurry = it.filename == "blur.png"

    result = auto_clean(
        store,
        remove_duplicates=True,
        fix_oob=True,
        remove_blurry=True,
        remove_empty=False,
        cache=HashCache(tmp_path / "c.json"),
    )
    assert result.kept == 1  # 只剩 a（或 b）
    assert result.removed == 2
