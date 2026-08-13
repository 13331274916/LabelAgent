"""清洗模块 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from labelagent.api.app import err, get_state, ok
from labelagent.config import DEFAULT_BLUR_THRESHOLD

router = APIRouter()


class DiagnoseBody(BaseModel):
    blur_threshold: float = Field(default=DEFAULT_BLUR_THRESHOLD, ge=0, le=10000)
    color_std_threshold: float = Field(default=3.0, ge=0, le=100)


class CleanBody(BaseModel):
    remove_duplicates: bool = True
    fix_oob: bool = True
    remove_blurry: bool = False
    remove_empty: bool = False
    duplicate_method: str = "lsh"


@router.post("/diagnose")
def diagnose(request: Request, body: DiagnoseBody):
    from labelagent.cleaning.bounds import check_out_of_bounds
    from labelagent.cleaning.duplicate import detect_duplicates
    from labelagent.cleaning.quality import diagnose as _diagnose

    state = get_state(request)
    if len(state.store) == 0:
        return err("请先在“标注”页面导入图像")

    result = _diagnose(state.store, blur_threshold=body.blur_threshold, color_std_threshold=body.color_std_threshold)

    # 重复图检测与越界检查合并进诊断报告
    images = state.store.list_images()
    dups = detect_duplicates(images, method="lsh", cache=state.hash_cache)
    result.duplicate_count = len(dups)
    oob = check_out_of_bounds(state.store)
    result.oob_count = sum(oob.values())
    result.anomalies = (
        result.blurry_count + result.empty_count + result.duplicate_count + result.oob_count
    )
    # 重新计算健康得分
    from labelagent.cleaning.quality import _health_score

    result.health_score = _health_score(result)
    return ok(result.model_dump())


@router.post("/auto-clean")
def auto_clean(request: Request, body: CleanBody):
    from labelagent.cleaning.cleaner import auto_clean as _auto_clean

    state = get_state(request)
    if len(state.store) == 0:
        return err("当前没有图像数据")
    result = _auto_clean(
        state.store,
        remove_duplicates=body.remove_duplicates,
        fix_oob=body.fix_oob,
        remove_blurry=body.remove_blurry,
        remove_empty=body.remove_empty,
        duplicate_method=body.duplicate_method,
        cache=state.hash_cache,
    )
    return ok(result.model_dump())


@router.get("/cache")
def cache_info(request: Request):
    state = get_state(request)
    return ok(
        {
            "file": str(state.hash_cache.path),
            "entries": len(state.hash_cache._data),
        }
    )


@router.delete("/cache")
def clear_cache(request: Request):
    state = get_state(request)
    n = state.hash_cache.clear()
    return ok({"cleared": n})
