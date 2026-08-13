"""环境模块 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from labelagent.api.app import err, get_state, ok

router = APIRouter()


class ImportEnvBody(BaseModel):
    path: str


@router.get("/detect")
def detect(request: Request):
    from labelagent.environment.python_env import detect_python_envs

    envs = detect_python_envs()
    # 只返回概要，避免返回全部依赖包
    return ok(
        [
            {
                "path": e.path,
                "version": e.version,
                "prefix": e.prefix,
                "valid": e.valid,
                "error": e.error,
                "package_count": len(e.packages),
            }
            for e in envs
        ]
    )


@router.post("/import")
def import_env(request: Request, body: ImportEnvBody):
    from labelagent.environment.python_env import import_python_env

    info = import_python_env(body.path)
    if not info.valid:
        return err(f"Python 环境无效：{info.error}")
    return ok(
        {
            "path": info.path,
            "version": info.version,
            "prefix": info.prefix,
            "package_count": len(info.packages),
        }
    )


@router.post("/packages")
def packages(request: Request, body: ImportEnvBody):
    """读取指定解释器的依赖包列表。"""
    from labelagent.environment.python_env import import_python_env

    info = import_python_env(body.path)
    if not info.valid:
        return err(f"Python 环境无效：{info.error}")
    return ok([p.model_dump() for p in info.packages])
