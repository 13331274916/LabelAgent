"""训练配置与脚本生成测试。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from labelagent.core.models import AblationGroup, TrainingConfig


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.epochs == 12
    assert cfg.batch == 8
    assert cfg.img_size == 640
    assert cfg.lr == 0.01
    assert cfg.two_stage is True
    assert cfg.stage1_epochs == 8
    assert cfg.stage2_epochs == 4


def test_training_config_validation():
    with pytest.raises(ValidationError):
        TrainingConfig(lr=-0.1)
    with pytest.raises(ValidationError):
        TrainingConfig(epochs=0)
    with pytest.raises(ValidationError):
        TrainingConfig(epochs=6, stage1_epochs=8)  # stage1 必须小于总 epochs


def test_effective_strategies():
    cfg = TrainingConfig(strategies={"warmup": True, "ema": True, "mosaic": False})
    names = cfg.effective_strategies
    assert "warmup" in names and "ema" in names
    assert "mosaic" not in names


def test_ablation_group_defaults():
    g = AblationGroup(name="baseline", model_arch="RT-DETR")
    assert g.warmup is False
    assert g.id


def test_scriptgen_contains_config():
    from labelagent.training.scriptgen import dump_config, generate_train_script

    cfg = TrainingConfig(
        model_arch="YOLOv8-Seg",
        loss="Focal Loss + EIoU",
        optimizer="AdamW",
        strategies={"ema": True},
    )
    script = generate_train_script(cfg)
    assert "results.csv" in script
    assert "best_stage" in script

    data = dump_config(cfg)
    assert data["model_arch"] == "YOLOv8-Seg"
    assert data["strategies"] == {"ema": True}
    # 应可序列化为 JSON（供外部脚本读取）
    json.dumps(data)
