"""多组消融实验：配置管理、批量运行、CSV 导入对比、结果导出与对比图。"""

from __future__ import annotations

import csv
import io
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from labelagent.core.models import (
    AblationGroup,
    AblationMetrics,
    TrainingConfig,
)
from labelagent.training.runner import TrainingRunner

_RESULT_FIELDS = ["dataset", "model_arch", "val_loss", "mAP@0.5", "precision", "note"]


class AblationManager:
    """管理消融实验组，并编排批量运行与结果对比。"""

    def __init__(self, runner: TrainingRunner) -> None:
        self._runner = runner
        self._groups: list[AblationGroup] = []
        self._results: list[AblationMetrics] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # 实验组管理
    # ------------------------------------------------------------------
    def add_group(self, group: AblationGroup) -> AblationGroup:
        with self._lock:
            if any(g.name == group.name for g in self._groups):
                raise ValueError(f"实验组名称已存在：{group.name}")
            self._groups.append(group)
            return group

    def remove_group(self, group_id: str) -> bool:
        with self._lock:
            before = len(self._groups)
            self._groups = [g for g in self._groups if g.id != group_id]
            return len(self._groups) < before

    def list_groups(self) -> list[AblationGroup]:
        with self._lock:
            return list(self._groups)

    def clear(self) -> None:
        with self._lock:
            self._groups.clear()
            self._results.clear()

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # 批量运行
    # ------------------------------------------------------------------
    def run_all(
        self,
        dataset_yaml: Optional[str],
        classes: list[str],
        mode: str = "demo",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        """在后台线程依次运行所有实验组。"""
        if self._running:
            raise RuntimeError("消融实验正在运行中")
        self._running = True
        self._thread = threading.Thread(
            target=self._run_all_sync,
            kwargs={"dataset_yaml": dataset_yaml, "classes": classes, "mode": mode, "on_log": on_log},
            daemon=True,
        )
        self._thread.start()

    def _run_all_sync(
        self,
        dataset_yaml: Optional[str],
        classes: list[str],
        mode: str,
        on_log: Optional[Callable[[str], None]],
    ) -> None:
        try:
            groups = self.list_groups()
            with self._lock:
                self._results = []
            for g in groups:
                if on_log:
                    on_log(f"开始实验组：{g.name}（{g.model_arch}）")
                config = TrainingConfig(
                    model_arch=g.model_arch,
                    loss=g.loss,
                    optimizer=g.optimizer,
                    scheduler=g.scheduler,
                    epochs=g.epochs,
                    batch=g.batch,
                    lr=g.lr,
                    two_stage=False,
                    strategies={
                        "warmup": g.warmup,
                        "ema": g.ema,
                        "mosaic": g.mosaic,
                    },
                    classes=classes,
                    variant_name=g.name,
                )
                self._runner.start(config, mode=mode, dataset_yaml=dataset_yaml)
                # 等待该组训练结束
                while self._runner.is_running():
                    time.sleep(0.2)
                info = self._runner.status()
                if info is None or info.status != "finished":
                    if on_log:
                        on_log(f"实验组 {g.name} 失败：{(info.error if info else 'unknown')}")
                    continue
                metrics = self._metrics_from_run(g, info)
                with self._lock:
                    self._results.append(metrics)
                if on_log:
                    on_log(f"实验组 {g.name} 完成：mAP@0.5={metrics.mAP50:.4f}")
        finally:
            self._running = False

    def _metrics_from_run(self, group: AblationGroup, info) -> AblationMetrics:
        """从一次训练运行的 results.csv 提取指标。"""
        val_loss, mAP50, precision = 0.0, 0.0, 0.0
        csv_path = info.artifacts.get("results.csv")
        if csv_path and Path(csv_path).exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                best = max(rows, key=lambda r: float(r.get("mAP@0.5", 0) or 0))
                val_loss = float(best.get("val_loss", 0) or 0)
                mAP50 = float(best.get("mAP@0.5", 0) or 0)
                precision = float(best.get("precision", 0) or 0)
        return AblationMetrics(
            group_id=group.id,
            name=group.name,
            model_arch=group.model_arch,
            val_loss=val_loss,
            mAP50=mAP50,
            precision=precision,
            note="demo" if info.mode == "demo" else "external",
        )

    # ------------------------------------------------------------------
    # CSV 导入
    # ------------------------------------------------------------------
    def import_results_csv(self, files: list[str], base_name: str = "external") -> list[AblationMetrics]:
        """批量导入已有 results.csv，加入当前对比。"""
        imported: list[AblationMetrics] = []
        for idx, path in enumerate(files, start=1):
            rows = _read_csv(path)
            if not rows:
                continue
            best = max(rows, key=lambda r: float(r.get("mAP@0.5", 0) or 0))
            name = f"{base_name}_{idx}" if len(files) > 1 else base_name
            imported.append(
                AblationMetrics(
                    group_id=f"ext_{idx}",
                    name=name,
                    model_arch=best.get("model_arch", "external"),
                    val_loss=float(best.get("val_loss", 0) or 0),
                    mAP50=float(best.get("mAP@0.5", 0) or 0),
                    precision=float(best.get("precision", 0) or 0),
                    note="imported",
                )
            )
        with self._lock:
            self._results.extend(imported)
        return imported

    # ------------------------------------------------------------------
    # 结果汇总
    # ------------------------------------------------------------------
    def results(self) -> list[AblationMetrics]:
        """返回带 best / 相对提升标记的结果列表。"""
        with self._lock:
            items = [r.model_copy() for r in self._results]
        if not items:
            return items
        baseline = items[0].mAP50 or 1e-9
        best_mAP = max(i.mAP50 for i in items)
        for it in items:
            it.is_best = abs(it.mAP50 - best_mAP) < 1e-9
            it.relative_improvement = round((it.mAP50 - baseline) / baseline * 100.0, 2)
        return items

    def to_summary_csv(self) -> str:
        """生成 ablation_summary.csv 文本。"""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["数据集/变体名称", "模型架构", "Val Loss", "mAP@0.5", "Precision", "相对提升(%)", "最佳"])
        for r in self.results():
            writer.writerow(
                [
                    r.name,
                    r.model_arch,
                    r.val_loss,
                    r.mAP50,
                    r.precision,
                    r.relative_improvement if r.relative_improvement is not None else "",
                    "★" if r.is_best else "",
                ]
            )
        return buf.getvalue()

    def export_summary(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_summary_csv(), encoding="utf-8")
        return str(path)

    def export_plot(self, path: str | Path) -> Optional[str]:
        """生成消融对比图（matplotlib），未安装时返回 None。"""
        items = self.results()
        if not items:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        names = [r.name for r in items]
        mAPs = [r.mAP50 for r in items]
        colors = ["#e74c3c" if r.is_best else "#3498db" for r in items]
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 4.5))
        ax.bar(names, mAPs, color=colors)
        ax.set_ylabel("mAP@0.5")
        ax.set_title("Ablation Comparison (mAP@0.5)")
        ax.set_ylim(0, min(1.0, max(mAPs) * 1.25))
        for i, v in enumerate(mAPs):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)


def _read_csv(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []
