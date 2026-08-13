"""FastAPI 应用工厂：挂载各模块路由与 Web 静态资源。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from labelagent.cleaning.duplicate import HashCache
from labelagent.config import WEB_DIR, ensure_workspace, setup_logging
from labelagent.core.store import ProjectStore
from labelagent.training.ablation import AblationManager
from labelagent.training.runner import TrainingRunner


class AppState:
    """应用级共享状态。"""

    def __init__(self) -> None:
        self.store = ProjectStore()
        self.hash_cache = HashCache()
        self.runner = TrainingRunner()
        self.ablation = AblationManager(self.runner)
        self.agent_annotator = None  # 延迟导入避免循环依赖


def create_app() -> FastAPI:
    logger = setup_logging()
    ensure_workspace()
    state = AppState()

    from labelagent.annotation.agent import AgentAnnotator

    state.agent_annotator = AgentAnnotator(state.store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("LabelAgent 服务启动")
        yield
        logger.info("LabelAgent 服务关闭")

    app = FastAPI(title="LabelAgent", version="0.1.0", lifespan=lifespan)
    app.state.la = state

    # 允许桌面 WebView 与开发端口跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 各模块路由
    from labelagent.api import (
        annotation_routes,
        cleaning_routes,
        dataset_routes,
        demo_routes,
        environment_routes,
        training_routes,
    )

    app.include_router(annotation_routes.router, prefix="/api/annotation")
    app.include_router(cleaning_routes.router, prefix="/api/cleaning")
    app.include_router(dataset_routes.router, prefix="/api/dataset")
    app.include_router(environment_routes.router, prefix="/api/environment")
    app.include_router(training_routes.router, prefix="/api/training")
    app.include_router(demo_routes.router, prefix="/api/demo")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "LabelAgent", "images": len(state.store)}

    @app.get("/api/overview")
    def overview():
        """前端首页概览数据。"""
        stats = state.store
        return {
            "images": len(stats),
            "annotations": sum(i.annotation_count for i in stats.list_images()),
            "classes": stats.class_list(),
        }

    # Web 静态资源（桌面版 WebView 加载同一套页面）
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


def download_response(path: str, filename: str | None = None) -> FileResponse:
    """构造文件下载响应（自动处理文件名编码）。"""
    return FileResponse(
        path,
        filename=filename,
        media_type="application/octet-stream",
    )


def ok(data) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data})


def err(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status)


# 兼容直接 import
def get_state(request) -> AppState:
    return request.app.state.la
