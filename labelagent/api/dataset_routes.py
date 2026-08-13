"""数据集模块 API。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from labelagent.api.app import download_response, err, get_state, ok
from labelagent.config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_SPLIT_RATIO,
    SPLIT_RATIO_MAX,
    SPLIT_RATIO_MIN,
    WORKSPACE_DIR,
)
from labelagent.core.models import SplitResult

router = APIRouter()


class SplitBody(BaseModel):
    train_ratio: float = Field(default=DEFAULT_SPLIT_RATIO, ge=SPLIT_RATIO_MIN, le=SPLIT_RATIO_MAX)
    dataset_name: str = DEFAULT_DATASET_NAME


@router.get("/stats")
def stats(request: Request):
    from labelagent.dataset.stats import compute_stats

    state = get_state(request)
    return ok(compute_stats(state.store).model_dump())


@router.post("/split")
def split(request: Request, body: SplitBody):
    from labelagent.dataset.split import split_dataset

    state = get_state(request)
    if len(state.store) == 0:
        return err("当前没有标注数据，请先在“标注”页面导入并完成标注")
    if not body.dataset_name.strip():
        return err("数据集名称不能为空")
    try:
        result: SplitResult = split_dataset(
            state.store,
            train_ratio=body.train_ratio,
            dataset_name=body.dataset_name.strip(),
        )
    except ValueError as e:
        return err(str(e))
    return ok(result.model_dump())


@router.get("/yaml")
def get_yaml(request: Request, path: str):
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "dataset.yaml 不存在")
    return download_response(str(p), filename=p.name)


@router.get("/datasets")
def list_datasets(request: Request):
    """列出工作目录下已生成的数据集。"""
    root = WORKSPACE_DIR / "datasets"
    if not root.is_dir():
        return ok([])
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            yaml_file = d / "dataset.yaml"
            out.append(
                {
                    "name": d.name,
                    "path": str(d),
                    "has_yaml": yaml_file.exists(),
                }
            )
    return ok(out)
