"""图像质量诊断：模糊检测（Laplacian 方差）、空图检测、健康得分。

模糊检测使用灰度图 Laplacian 算子方差作为锐度指标，低于阈值判定为模糊。
阈值对应桌面版 README 中记录的 cleanBlurThresh（默认 100）。
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PIL import Image

from labelagent.config import DEFAULT_BLUR_THRESHOLD
from labelagent.core.models import DiagnosticResult
from labelagent.core.store import ProjectStore

# 各异常项的健康扣分
PENALTY = {"blurry": 5.0, "duplicate": 8.0, "oob": 5.0, "empty": 8.0}


def laplacian_variance(image: Image.Image) -> float:
    """计算灰度图的 Laplacian 方差（越大越清晰）。

    使用离散 Laplacian 算子（4-邻域）直接对灰度矩阵卷积，
    纯色/严重模糊图像方差接近 0，纹理清晰图像方差较大。
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    padded = np.pad(arr, 1, mode="edge")
    # lap = 4 邻域 Laplacian：up + down + left + right - 4*center
    lap = (
        padded[1:-1, 2:]
        + padded[1:-1, :-2]
        + padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(lap.var())


def is_empty_image(image: Image.Image, color_std_threshold: float = 3.0) -> bool:
    """判断是否为“空图”：近似纯色/无有效内容。"""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    if arr.size == 0:
        return True
    return float(arr.std()) < color_std_threshold


def diagnose(
    store: ProjectStore,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    color_std_threshold: float = 3.0,
) -> DiagnosticResult:
    """对项目内全部图像执行质量诊断，更新图像标记并返回汇总结果。"""
    start = time.time()
    images = store.list_images()
    result = DiagnosticResult(scanned=len(images))

    details: dict[str, dict] = {}
    for item in images:
        flags = {"blurry": False, "empty": False}
        try:
            with Image.open(item.path) as im:
                var = laplacian_variance(im)
                flags["blurry"] = var < blur_threshold
                flags["empty"] = is_empty_image(im, color_std_threshold)
        except OSError:
            # 文件不可读按空图处理
            flags["empty"] = True

        item.is_blurry = flags["blurry"]
        item.is_empty = flags["empty"]
        if flags["blurry"]:
            result.blurry_count += 1
        if flags["empty"]:
            result.empty_count += 1
        details[item.id] = {"filename": item.filename, **flags, "laplacian_var": round(var, 2)}

    result.details = details
    result.anomalies = result.blurry_count + result.empty_count
    result.elapsed_ms = round((time.time() - start) * 1000, 1)
    result.health_score = _health_score(result)
    return result


def _health_score(result: DiagnosticResult) -> float:
    if result.scanned == 0:
        return 100.0
    score = 100.0
    score -= PENALTY["blurry"] * result.blurry_count
    score -= PENALTY["duplicate"] * result.duplicate_count
    score -= PENALTY["oob"] * result.oob_count
    score -= PENALTY["empty"] * result.empty_count
    return round(max(0.0, min(100.0, score)), 1)
