"""标注模块 API。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from labelagent.api.app import download_response, err, get_state, ok
from labelagent.config import EXPORT_DIR, WORKSPACE_DIR

router = APIRouter()


class ImportFolderBody(BaseModel):
    folder: str
    recursive: bool = True


class ImportPathsBody(BaseModel):
    paths: list[str]


class AgentStartBody(BaseModel):
    provider_id: str = "demo"
    prompt: str = "标注汽车, 划痕, 缺陷"
    label: str = "scratch_defect"
    max_objects: int = Field(default=5, ge=1, le=100)
    api_key: str | None = None
    image_ids: list[str] | None = None


class AnnotationBody(BaseModel):
    label: str = "scratch_defect"
    points: list[dict] | None = None
    confidence: float | None = None


class UpdateAnnotationBody(BaseModel):
    label: str | None = None
    points: list[dict] | None = None
    confidence: float | None = None


class ExportSingleBody(BaseModel):
    format: str


class ExportZipBody(BaseModel):
    formats: list[str] = ["labelme", "voc", "yolo", "coco", "csv"]
    filename: str | None = None


@router.get("/providers")
def providers(request: Request):
    from labelagent.annotation.providers import list_providers

    return ok(list_providers())


@router.post("/import-folder")
def import_folder(request: Request, body: ImportFolderBody):
    from labelagent.annotation.importer import import_folder as _import_folder

    state = get_state(request)
    if not Path(body.folder).is_dir():
        return err(f"文件夹不存在：{body.folder}")
    items = _import_folder(body.folder, state.store, recursive=body.recursive)
    return ok({"imported": len(items), "total": len(state.store)})


@router.post("/import-paths")
def import_paths(request: Request, body: ImportPathsBody):
    from labelagent.annotation.importer import import_paths as _import_paths

    state = get_state(request)
    items = _import_paths(body.paths, state.store)
    return ok({"imported": len(items), "total": len(state.store)})


@router.post("/upload")
async def upload(request: Request, files: list[UploadFile] = File(...)):
    """通过浏览器上传图像（Web 模式下导入本地图片）。"""
    state = get_state(request)
    upload_dir = WORKSPACE_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if not f.filename:
            continue
        target = upload_dir / f.filename
        content = await f.read()
        target.write_bytes(content)
        saved.append(str(target))
    from labelagent.annotation.importer import import_paths as _import_paths

    items = _import_paths(saved, state.store)
    return ok({"imported": len(items), "total": len(state.store)})


@router.get("/images")
def images(request: Request):
    state = get_state(request)
    return ok([it.stat() for it in state.store.list_images()])


@router.get("/image/{image_id}")
def image_detail(request: Request, image_id: str):
    state = get_state(request)
    item = state.store.get(image_id)
    if item is None:
        raise HTTPException(404, "图像不存在")
    return ok(item.model_dump())


@router.get("/image/{image_id}/file")
def image_file(request: Request, image_id: str):
    state = get_state(request)
    item = state.store.get(image_id)
    if item is None or not Path(item.path).is_file():
        raise HTTPException(404, "图像文件不存在")
    return download_response(item.path, filename=item.filename)


@router.post("/agent/start")
def agent_start(request: Request, body: AgentStartBody):
    state = get_state(request)
    try:
        status = state.agent_annotator.start(
            provider_id=body.provider_id,
            prompt=body.prompt,
            label=body.label,
            max_objects=body.max_objects,
            api_key=body.api_key,
            image_ids=body.image_ids,
        )
    except (KeyError, RuntimeError) as e:
        return err(str(e))
    return ok(status.model_dump())


@router.get("/agent/status")
def agent_status(request: Request):
    state = get_state(request)
    return ok(state.agent_annotator.status.model_dump())


@router.post("/image/{image_id}/annotations")
def add_annotation(request: Request, image_id: str, body: AnnotationBody):
    from labelagent.annotation.editor import add_annotation as _add

    state = get_state(request)
    ann = _add(state.store, image_id, body.label, body.points or [])
    if ann is None:
        raise HTTPException(404, "图像不存在")
    return ok(ann.model_dump())


@router.patch("/image/{image_id}/annotations/{ann_id}")
def update_annotation(request: Request, image_id: str, ann_id: str, body: UpdateAnnotationBody):
    state = get_state(request)
    ann = state.store.update_annotation(
        image_id,
        ann_id,
        label=body.label,
        points=body.points,
        confidence=body.confidence,
    )
    if ann is None:
        raise HTTPException(404, "标注不存在")
    return ok(ann.model_dump())


@router.delete("/image/{image_id}/annotations/{ann_id}")
def delete_annotation(request: Request, image_id: str, ann_id: str):
    state = get_state(request)
    deleted = state.store.delete_annotation(image_id, ann_id)
    if not deleted:
        raise HTTPException(404, "标注不存在")
    return ok({"deleted": True})


@router.post("/image/{image_id}/export")
def export_single(request: Request, image_id: str, body: ExportSingleBody):
    from labelagent.annotation.export import export_single as _export

    state = get_state(request)
    item = state.store.get(image_id)
    if item is None:
        raise HTTPException(404, "图像不存在")
    try:
        path = _export(item, body.format, out_dir=EXPORT_DIR)
    except ValueError as e:
        return err(str(e))
    return ok({"path": path, "filename": Path(path).name})


@router.post("/export-zip")
def export_zip(request: Request, body: ExportZipBody):
    from labelagent.annotation.export import export_zip as _export_zip

    state = get_state(request)
    images = state.store.list_images()
    if not images:
        return err("当前没有可导出的图像")
    filename = body.filename or "labelme_dataset_project.zip"
    out_zip = EXPORT_DIR / filename
    try:
        path = _export_zip(images, body.formats, out_zip=out_zip)
    except ValueError as e:
        return err(str(e))
    return ok({"path": path, "filename": Path(path).name})


@router.get("/download")
def download(request: Request, path: str):
    """下载导出文件（仅允许工作目录内路径，防止目录穿越）。"""
    resolved = Path(path).resolve()
    root = Path(WORKSPACE_DIR).resolve()
    if not str(resolved).startswith(str(root)):
        raise HTTPException(403, "路径超出工作目录")
    if not resolved.is_file():
        raise HTTPException(404, "文件不存在")
    return download_response(str(resolved), filename=resolved.name)
