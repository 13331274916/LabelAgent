"""演示模块 API：生成合成演示图像，便于无真实数据时体验全流程。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from labelagent.api.app import get_state, ok
from labelagent.demo import generate_demo_images

router = APIRouter()


class GenerateBody(BaseModel):
    count: int = Field(default=12, ge=1, le=200)
    width: int = Field(default=640, ge=64, le=2048)
    height: int = Field(default=480, ge=64, le=2048)


@router.post("/generate")
def generate(request: Request, body: GenerateBody):
    """生成 count 张合成图像（含形状、噪声、模糊/重复样本）并导入项目。"""
    state = get_state(request)
    items = generate_demo_images(
        state.store,
        count=body.count,
        width=body.width,
        height=body.height,
    )
    return ok({"generated": len(items), "total": len(state.store)})
