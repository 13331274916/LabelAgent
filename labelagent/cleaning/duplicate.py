"""重复图检测：感知哈希（dHash）+ LSH 分桶 / Pairwise 精确匹配 + 哈希缓存。

- dHash：对图像缩放到 9x8 灰度，比较相邻像素明暗得到 64 位哈希；
- LSH 分桶匹配（O(N) 候选）：按哈希的不同位段分桶，桶内做汉明距离精确比较；
- Pairwise 精确匹配（O(N²)）：全量两两比较，适合小数据集；
- 哈希缓存：将计算结果持久化到 workspace/cache/image_hashes.json，
  按「路径 + 文件大小 + 修改时间」判断缓存是否仍然有效。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from labelagent.config import HASH_CACHE_FILE
from labelagent.core.models import ImageItem

HAMMING_THRESHOLD = 6  # 汉明距离 <= 该值判定为重复


# ---------------------------------------------------------------------------
# dHash
# ---------------------------------------------------------------------------
def dhash(path: str, hash_size: int = 8) -> Optional[int]:
    """计算图像 dHash 64 位整数；失败返回 None。"""
    try:
        with Image.open(path) as im:
            gray = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    except OSError:
        return None
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, :-1] > arr[:, 1:]  # 每行相邻像素比较
    bits = diff.flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# 哈希缓存
# ---------------------------------------------------------------------------
class HashCache:
    """图像哈希持久化缓存（JSON 文件）。"""

    def __init__(self, path: str | Path = HASH_CACHE_FILE) -> None:
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")

    def get(self, path: str, size: int, mtime: float) -> Optional[int]:
        entry = self._data.get(path)
        if entry and entry.get("size") == size and abs(entry.get("mtime", 0) - mtime) < 1:
            return entry.get("hash")
        return None

    def put(self, path: str, size: int, mtime: float, hash_value: int) -> None:
        self._data[path] = {"size": size, "mtime": mtime, "hash": hash_value}

    def clear(self) -> int:
        n = len(self._data)
        self._data = {}
        if self.path.exists():
            self.path.unlink()
        return n


# ---------------------------------------------------------------------------
# 检测入口
# ---------------------------------------------------------------------------
def _file_meta(path: str) -> tuple[int, float]:
    try:
        st = Path(path).stat()
        return st.st_size, st.st_mtime
    except OSError:
        return 0, 0.0


def detect_duplicates(
    images: list[ImageItem],
    method: str = "lsh",
    cache: Optional[HashCache] = None,
    threshold: int = HAMMING_THRESHOLD,
) -> dict[str, str]:
    """检测重复图，返回 {被判定为重复的图像 id: 保留的第一张图像 id}。

    method: "lsh" 分桶匹配（大数据集） / "pairwise" 精确匹配（小数据集）。
    """
    use_cache = cache is not None
    hashes: dict[str, int] = {}
    for item in images:
        h = None
        if use_cache:
            size, mtime = _file_meta(item.path)
            h = cache.get(item.path, size, mtime)
        if h is None:
            h = dhash(item.path)
            if h is not None and use_cache:
                size, mtime = _file_meta(item.path)
                cache.put(item.path, size, mtime, h)
        if h is not None:
            hashes[item.id] = h
    if use_cache:
        cache.save()

    duplicates: dict[str, str] = {}
    if method == "pairwise":
        ids = list(hashes.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if hamming_distance(hashes[ids[i]], hashes[ids[j]]) <= threshold:
                    duplicates[ids[j]] = ids[i]
    else:  # lsh：按 8 位段分桶
        buckets: dict[int, list[str]] = {}
        for image_id, h in hashes.items():
            for seg in range(0, 64, 8):
                key = (h >> seg) & 0xFF
                buckets.setdefault(key * 256 + seg, []).append(image_id)
        seen_kept: dict[str, str] = {}
        for bucket in buckets.values():
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    if hamming_distance(hashes[a], hashes[b]) <= threshold:
                        kept = seen_kept.get(a, a)
                        duplicates[b] = kept
                        seen_kept[b] = kept

    # 去重并更新图像标记
    for image_id, dup_of in duplicates.items():
        for item in images:
            if item.id == image_id:
                item.duplicate_of = dup_of
            elif item.id == dup_of and item.duplicate_of is None:
                item.duplicate_of = None
    return duplicates
