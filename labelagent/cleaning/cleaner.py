"""自动清洗编排：剔除重复图 / 纠偏越界框 / 剔除模糊图 / 剔除空图。"""

from __future__ import annotations

from typing import Optional

from labelagent.cleaning.bounds import fix_out_of_bounds
from labelagent.cleaning.duplicate import HashCache, detect_duplicates
from labelagent.core.models import CleanResult
from labelagent.core.store import ProjectStore


def auto_clean(
    store: ProjectStore,
    remove_duplicates: bool = True,
    fix_oob: bool = True,
    remove_blurry: bool = False,
    remove_empty: bool = False,
    duplicate_method: str = "lsh",
    cache: Optional[HashCache] = None,
) -> CleanResult:
    """执行自动清洗，返回保留数量与修复统计。"""
    result = CleanResult()
    images = store.list_images()

    # 1. 剔除重复图
    if remove_duplicates:
        dups = detect_duplicates(images, method=duplicate_method, cache=cache)
        dup_ids = list(dups.keys())
        removed = store.remove_many(dup_ids)
        result.removed += removed
        result.removed_ids.extend(dup_ids)
        # 被移除图像从其余图像的 duplicate_of 标记中清除
        for item in store.list_images():
            if item.duplicate_of in dup_ids:
                item.duplicate_of = None

    # 2. 纠偏越界框
    if fix_oob:
        result.fixed_ids = fix_out_of_bounds(store)
        result.fixed_oob = len(result.fixed_ids)

    # 3/4. 剔除模糊图 / 空图
    for item in store.list_images():
        if (remove_blurry and item.is_blurry) or (remove_empty and item.is_empty):
            store.remove(item.id)
            result.removed += 1
            result.removed_ids.append(item.id)

    result.kept = len(store)
    return result
