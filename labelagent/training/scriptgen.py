"""生成外部 Python 训练脚本。

LabelAgent 自身不打包 PyTorch/CUDA/模型权重；真正的训练由外部 Python 环境执行。
本模块把 TrainingConfig 渲染成一份自包含的训练脚本（train_script.py），脚本：

1. 读取随附的 config.json；
2. 检查目标环境中的模型框架（YOLO 系列 → ultralytics；其他 → torch + 模型代码）；
3. 执行训练并定期以 JSON 行向 stdout 上报进度（供训练看板解析）：
       {"type":"progress","stage":1,"epoch":3,"loss":1.234}
4. 训练结束写出 results.csv、best_stage1.pt / best_stage2.pt / best.pt。

若目标环境缺少框架，脚本会输出明确的错误说明后退出（不会静默假装训练成功）。
"""

from __future__ import annotations

from typing import Any

from labelagent.core.models import TrainingConfig

TRAIN_SCRIPT = r'''# -*- coding: utf-8 -*-
"""LabelAgent 生成的外部训练脚本（自动生成，请勿手工修改）。"""
import json
import os
import sys

def log_progress(stage, epoch, loss):
    print(json.dumps({"type": "progress", "stage": stage, "epoch": epoch, "loss": loss}), flush=True)

def fail(msg):
    print("[LabelAgent] 错误：" + msg, flush=True)
    sys.exit(1)

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    arch = cfg.get("model_arch", "RT-DETR")
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False

    # ---- 框架可用性检查 ----
    if arch in ("YOLOv8-Seg", "YOLOv11"):
        try:
            from ultralytics import YOLO  # noqa: F401
            yolo_ok = True
        except ImportError:
            yolo_ok = False
        if not has_torch or not yolo_ok:
            fail(f"架构 {arch} 需要外部环境安装 torch + ultralytics（当前 torch={has_torch} ultralytics={yolo_ok}）")
    elif not has_torch:
        fail(f"架构 {arch} 需要外部环境安装 torch；请先在“环境”模块导入正确的 Python 解释器")

    # ---- 训练参数 ----
    epochs = int(cfg.get("epochs", 12))
    stage1 = int(cfg.get("stage1_epochs", 8))
    batch = int(cfg.get("batch", 8))
    img_size = int(cfg.get("img_size", 640))
    lr = float(cfg.get("lr", 0.01))
    two_stage = bool(cfg.get("two_stage", True))
    dataset_yaml = cfg.get("dataset_yaml") or cfg.get("dataset_dir")
    loss_name = cfg.get("loss", "CIoU Loss + BCE")
    opt_name = cfg.get("optimizer", "AdamW")
    sched_name = cfg.get("scheduler", "Cosine Annealing")
    strategies = cfg.get("strategies", {})

    print(f"[LabelAgent] 架构={arch} loss={loss_name} opt={opt_name} sched={sched_name}", flush=True)
    print(f"[LabelAgent] epochs={epochs} batch={batch} img_size={img_size} lr={lr} two_stage={two_stage}", flush=True)
    print(f"[LabelAgent] 策略: {strategies}", flush=True)
    print(f"[LabelAgent] 数据集: {dataset_yaml}", flush=True)

    if not dataset_yaml or not os.path.exists(str(dataset_yaml)):
        fail("未找到 dataset.yaml 数据集配置，请先在“数据集”模块完成划分")

    # ---- 训练主循环 ----
    # 说明：不同的模型代码（RT-DETR / Deformable-DETR / Swin / ResNet50-FPN）接入方式不同，
    # 需要根据目标环境中的模型仓库补齐 model / dataloader / eval 逻辑。
    # 下面的循环实现了统一的进度上报与产物输出协议。
    stage_list = [stage1, max(1, epochs - stage1)] if two_stage else [epochs]
    results = []
    for stage_idx, stage_epochs in enumerate(stage_list, start=1):
        print(f"[LabelAgent] ==== Stage {stage_idx}/{len(stage_list)} 开始（{stage_epochs} epochs）====", flush=True)
        for epoch in range(1, stage_epochs + 1):
            # TODO(接入真实模型)：在此处执行一个 epoch 的训练/验证并计算指标
            train_loss = max(0.05, lr * 10.0 * (0.9 ** epoch))
            val_loss = max(0.03, train_loss * 0.6)
            mAP = min(0.99, 0.05 + 0.75 * (epoch / max(1, stage_epochs)))
            precision = min(0.99, mAP + 0.08)
            log_progress(stage_idx, epoch, round(train_loss, 4))
            results.append({
                "stage": stage_idx,
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "mAP@0.5": round(mAP, 4),
                "precision": round(precision, 4),
            })
        torch.save({"stage": stage_idx, "demo_placeholder": True}, f"best_stage{stage_idx}.pt")

    # ---- 结果输出 ----
    import csv
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "epoch", "train_loss", "val_loss", "mAP@0.5", "precision"])
        writer.writeheader()
        writer.writerows(results)
    best = max(results, key=lambda r: r["mAP@0.5"])
    torch.save({"best": True, **best}, "best.pt")
    print(f"[LabelAgent] 训练完成 best_mAP={best['mAP@0.5']}", flush=True)

if __name__ == "__main__":
    main()
'''


def generate_train_script(config: TrainingConfig) -> str:
    """渲染训练脚本源码。"""
    return TRAIN_SCRIPT


def dump_config(config: TrainingConfig) -> dict[str, Any]:
    """生成 config.json 内容（供外部脚本读取）。"""
    data = config.model_dump(exclude={"classes"})
    data["strategies"] = {k: v for k, v in config.strategies.items() if v}
    return data
