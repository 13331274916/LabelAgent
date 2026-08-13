"""训练与消融模块 API。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from pydantic import BaseModel, Field

from labelagent.api.app import download_response, err, get_state, ok
from labelagent.config import RUNS_DIR, WORKSPACE_DIR
from labelagent.core.models import AblationGroup, TrainingConfig

router = APIRouter()


# ---------------------------------------------------------------------------
# 训练配置与执行
# ---------------------------------------------------------------------------
@router.get("/options")
def options(request: Request):
    from labelagent.training.config import options as _options

    return ok(_options())


class StartTrainingBody(BaseModel):
    config: TrainingConfig
    mode: str = "demo"  # demo / external
    dataset_yaml: str | None = None


@router.post("/start")
def start_training(request: Request, body: StartTrainingBody):
    state = get_state(request)
    if body.mode not in ("demo", "external"):
        return err("mode 只能是 demo 或 external")
    try:
        info = state.runner.start(body.config, mode=body.mode, dataset_yaml=body.dataset_yaml)
    except RuntimeError as e:
        return err(str(e))
    return ok(info.model_dump())


@router.get("/status")
def training_status(request: Request):
    state = get_state(request)
    info = state.runner.status()
    return ok(info.model_dump() if info else None)


@router.post("/stop")
def stop_training(request: Request):
    state = get_state(request)
    state.runner.stop()
    return ok({"stopped": True})


@router.get("/artifacts")
def artifacts(request: Request):
    """返回最近一次运行可下载的产物。"""
    state = get_state(request)
    info = state.runner.status()
    files = []
    if info and info.artifacts:
        for name, path in info.artifacts.items():
            p = Path(path)
            if p.exists():
                files.append({"name": name, "path": str(p), "size": p.stat().st_size})
    return ok(files)


# ---------------------------------------------------------------------------
# 消融实验
# ---------------------------------------------------------------------------
@router.get("/ablation/groups")
def ablation_groups(request: Request):
    state = get_state(request)
    return ok([g.model_dump() for g in state.ablation.list_groups()])


@router.post("/ablation/groups")
def ablation_add_group(request: Request, body: AblationGroup):
    state = get_state(request)
    try:
        g = state.ablation.add_group(body)
    except ValueError as e:
        return err(str(e))
    return ok(g.model_dump())


@router.delete("/ablation/groups/{group_id}")
def ablation_remove_group(request: Request, group_id: str):
    state = get_state(request)
    removed = state.ablation.remove_group(group_id)
    if not removed:
        return err("实验组不存在", 404)
    return ok({"removed": True})


class AblationRunBody(BaseModel):
    mode: str = "demo"
    dataset_yaml: str | None = None


@router.post("/ablation/run")
def ablation_run(request: Request, body: AblationRunBody):
    state = get_state(request)
    groups = state.ablation.list_groups()
    if not groups:
        return err("请先新增至少一组消融实验")
    try:
        state.ablation.run_all(
            dataset_yaml=body.dataset_yaml,
            classes=state.store.class_list(),
            mode=body.mode,
            on_log=lambda line: None,
        )
    except RuntimeError as e:
        return err(str(e))
    return ok({"running": True, "groups": len(groups)})


@router.get("/ablation/status")
def ablation_status(request: Request):
    state = get_state(request)
    return ok(
        {
            "running": state.ablation.running,
            "groups": [g.model_dump() for g in state.ablation.list_groups()],
            "results": [r.model_dump() for r in state.ablation.results()],
        }
    )


@router.post("/ablation/import-csv")
async def ablation_import_csv(request: Request, files: list[UploadFile] = File(...), base_name: str = "external"):
    state = get_state(request)
    tmp = Path(tempfile.mkdtemp(prefix="ablation_import_"))
    paths = []
    for f in files:
        if not f.filename:
            continue
        p = tmp / f.filename
        p.write_bytes(await f.read())
        paths.append(str(p))
    if not paths:
        return err("未收到 CSV 文件")
    imported = state.ablation.import_results_csv(paths, base_name=base_name)
    return ok({"imported": len(imported), "results": len(state.ablation.results())})


@router.get("/ablation/summary.csv")
def ablation_summary_csv(request: Request):
    from fastapi.responses import Response

    state = get_state(request)
    csv_text = state.ablation.to_summary_csv()
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="ablation_summary.csv"'},
    )


@router.get("/ablation/plot.png")
def ablation_plot(request: Request):
    state = get_state(request)
    plot_path = state.ablation.export_plot(RUNS_DIR / "ablation_comparison.png")
    if plot_path is None:
        return err("当前没有结果或未安装 matplotlib")
    return download_response(plot_path, filename="ablation_comparison.png")


@router.get("/ablation/summary-file")
def ablation_summary_file(request: Request):
    """将汇总结果写入工作目录并返回路径（便于前端提供下载链接）。"""
    state = get_state(request)
    path = state.ablation.export_summary(WORKSPACE_DIR / "exports" / "ablation_summary.csv")
    return ok({"path": path})


# ---------------------------------------------------------------------------
# 通用下载
# ---------------------------------------------------------------------------
@router.get("/download")
def download(request: Request, path: str):
    resolved = Path(path).resolve()
    root = Path(WORKSPACE_DIR).resolve()
    if not str(resolved).startswith(str(root)):
        from fastapi import HTTPException

        raise HTTPException(403, "路径超出工作目录")
    if not resolved.is_file():
        from fastapi import HTTPException

        raise HTTPException(404, "文件不存在")
    return download_response(str(resolved), filename=resolved.name)
