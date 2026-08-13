"""训练执行器：演示模式（内置模拟训练）与外部模式（子进程调用外部 Python）。"""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from labelagent.config import RUNS_DIR
from labelagent.core.models import TrainingConfig, TrainingRunInfo
from labelagent.training.monitor import TrainingMonitor
from labelagent.training.scriptgen import dump_config, generate_train_script

# 结果表列
RESULT_COLUMNS = ["stage", "epoch", "train_loss", "val_loss", "mAP@0.5", "precision"]


class TrainingRunner:
    """管理一次训练运行。mode="demo" 时在本进程内模拟训练（无需 GPU/外部环境）。"""

    def __init__(self, run_dir: str | Path = RUNS_DIR) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._monitor: Optional[TrainingMonitor] = None
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @property
    def monitor(self) -> Optional[TrainingMonitor]:
        return self._monitor

    def status(self) -> Optional[TrainingRunInfo]:
        """返回当前运行状态（无运行记录时返回 None）。"""
        m = self._monitor
        return m.info if m else None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def start(self, config: TrainingConfig, mode: str = "demo", dataset_yaml: Optional[str] = None) -> TrainingRunInfo:
        """启动训练（非阻塞）。mode: demo / external。"""
        if self.is_running():
            raise RuntimeError("已有训练任务正在运行")
        self._stop_flag.clear()
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{random.randint(100, 999)}"
        run_dir = self.run_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        total_epochs = config.epochs
        self._monitor = TrainingMonitor(run_id=run_id, total_epochs=total_epochs, mode=mode)
        self._monitor.set_status("running")
        self._monitor.log(
            f"启动训练：mode={mode} arch={config.model_arch} loss={config.loss} "
            f"opt={config.optimizer} sched={config.scheduler} epochs={config.epochs}"
        )
        if config.effective_strategies:
            self._monitor.log(f"策略: {', '.join(config.effective_strategies.values())}")

        self._thread = threading.Thread(
            target=self._run,
            args=(config, mode, dataset_yaml, run_dir),
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def stop(self) -> None:
        """停止训练。"""
        self._stop_flag.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    # ------------------------------------------------------------------
    def _run(
        self,
        config: TrainingConfig,
        mode: str,
        dataset_yaml: Optional[str],
        run_dir: Path,
    ) -> None:
        monitor = self._monitor
        assert monitor is not None
        try:
            if mode == "external":
                self._run_external(config, dataset_yaml, run_dir, monitor)
            else:
                self._run_demo(config, run_dir, monitor)
        except Exception as e:  # noqa: BLE001
            monitor.log(f"训练异常：{e}")
            monitor.set_status("failed", error=str(e))

    def _run_demo(self, config: TrainingConfig, run_dir: Path, monitor: TrainingMonitor) -> None:
        """演示模式：确定性模拟两阶段训练，产出结果表与占位权重。

        随机种子由「变体名称 + 模型架构 + 策略」派生，使不同消融组得到
        不同的（可复现的）演示指标，便于观察对比效果。
        """
        variant_seed = f"{config.variant_name or 'train'}|{config.model_arch}|{sorted(config.strategies.items())}"
        rng = random.Random(2026 + sum(ord(c) for c in variant_seed))
        stages = (
            [(1, config.stage1_epochs), (2, config.stage2_epochs)] if config.two_stage else [(1, config.epochs)]
        )
        rows: list[dict] = []
        epoch_global = 0
        for stage, n_epochs in stages:
            monitor.log(f"==== Stage {stage}/{len(stages)} 开始（{n_epochs} epochs）====")
            base_loss = 2.0 if stage == 1 else 0.9
            for e in range(1, n_epochs + 1):
                if self._stop_flag.is_set():
                    monitor.log("训练被用户停止")
                    monitor.set_status("stopped")
                    self._write_artifacts(run_dir, rows)
                    return
                decay = math.exp(-0.35 * e)
                noise = rng.uniform(-0.02, 0.02)
                train_loss = max(0.01, base_loss * decay + noise)
                val_loss = max(0.01, train_loss * (0.65 + rng.uniform(-0.05, 0.05)))
                mAP = min(0.98, 0.06 + 0.80 * (e / max(1, n_epochs)) - rng.uniform(0, 0.02))
                precision = min(0.98, mAP + 0.06)

                epoch_global += 1
                monitor.update_epoch(epoch_global, train_loss, stage=stage)
                monitor.log(
                    f"[Stage {stage}] epoch {e}/{n_epochs} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} mAP@0.5={mAP:.4f}"
                )
                rows.append(
                    {
                        "stage": stage,
                        "epoch": e,
                        "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "mAP@0.5": round(mAP, 4),
                        "precision": round(precision, 4),
                    }
                )
                time.sleep(0.25)
        self._write_artifacts(run_dir, rows)
        monitor.log("训练完成（演示模式）")
        monitor.set_status("finished")

    def _write_artifacts(self, run_dir: Path, rows: list[dict]) -> None:
        """写出 results.csv 与占位权重文件。"""
        csv_path = run_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        artifacts: dict[str, str] = {"results.csv": str(csv_path)}
        for stage in sorted({r["stage"] for r in rows}):
            name = f"best_stage{stage}.pt"
            p = run_dir / name
            p.write_text(
                json.dumps(
                    {
                        "artifact": name,
                        "demo": True,
                        "note": "演示模式占位文件，非真实 PyTorch 权重。接入外部训练环境后将由训练脚本生成真实权重。",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            artifacts[name] = str(p)
        best = max(rows, key=lambda r: r["mAP@0.5"]) if rows else {}
        best_path = run_dir / "best.pt"
        best_path.write_text(
            json.dumps(
                {
                    "artifact": "best.pt",
                    "demo": True,
                    "best": best,
                    "note": "演示模式占位文件，非真实 PyTorch 权重。",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts["best.pt"] = str(best_path)
        self._monitor.set_artifacts(artifacts)

    # ------------------------------------------------------------------
    def _run_external(
        self,
        config: TrainingConfig,
        dataset_yaml: Optional[str],
        run_dir: Path,
        monitor: TrainingMonitor,
    ) -> None:
        """外部模式：生成脚本 + config.json，调用外部 Python 执行。"""
        python_exe = config.python_exe
        if not python_exe:
            monitor.log("外部模式需要指定 Python 解释器（python_exe），请先在“环境”模块导入")
            monitor.set_status("failed", error="缺少 python_exe")
            return

        script = run_dir / "train_script.py"
        cfg_json = run_dir / "config.json"
        script.write_text(generate_train_script(config), encoding="utf-8")
        data = dump_config(config)
        data["dataset_yaml"] = dataset_yaml
        cfg_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        monitor.log(f"调用外部 Python：{python_exe}")
        try:
            self._proc = subprocess.Popen(
                [python_exe, str(script), str(cfg_json)],
                cwd=str(run_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as e:
            monitor.set_status("failed", error=str(e))
            return

        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop_flag.is_set():
                break
            line = line.rstrip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if isinstance(msg, dict) and msg.get("type") == "progress":
                    monitor.update_epoch(int(msg["epoch"]), float(msg["loss"]), stage=int(msg.get("stage", 1)))
                else:
                    monitor.log(line)
            except (json.JSONDecodeError, ValueError):
                monitor.log(line)

        rc = self._proc.wait() if self._proc.poll() is None else self._proc.poll()
        self._proc = None

        if self._stop_flag.is_set():
            monitor.set_status("stopped")
            return
        if rc != 0:
            monitor.set_status("failed", error=f"外部训练进程退出码 {rc}")
            return

        # 收集产物
        artifacts: dict[str, str] = {}
        for name in ("results.csv", "best.pt", "best_stage1.pt", "best_stage2.pt"):
            p = run_dir / name
            if p.exists():
                artifacts[name] = str(p)
        monitor.set_artifacts(artifacts)
        monitor.set_status("finished")
