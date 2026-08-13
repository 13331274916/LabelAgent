"""训练配置：模型 / Loss / 优化器 / 调度器 / 训练策略的可选项与校验。"""

from __future__ import annotations

from labelagent.config import (
    LOSS_FUNCTIONS,
    MODEL_ARCHS,
    OPTIMIZERS,
    SCHEDULERS,
    STRATEGIES,
)

__all__ = [
    "LOSS_FUNCTIONS",
    "MODEL_ARCHS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "STRATEGIES",
    "options",
]


def options() -> dict:
    """返回前端可用的全部训练选项。"""
    return {
        "model_archs": MODEL_ARCHS,
        "loss_functions": LOSS_FUNCTIONS,
        "optimizers": OPTIMIZERS,
        "schedulers": SCHEDULERS,
        "strategies": STRATEGIES,
        "defaults": {
            "epochs": 12,
            "batch": 8,
            "img_size": 640,
            "lr": 0.01,
            "stage1_epochs": 8,
            "stage2_epochs": 4,
            "two_stage": True,
        },
    }
