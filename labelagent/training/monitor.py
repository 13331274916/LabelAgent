"""训练看板状态：进度、Loss 曲线、控制台日志（线程安全）。"""

from __future__ import annotations

import threading
import time
from typing import Optional

from labelagent.core.models import TrainingRunInfo

MAX_LOG_LINES = 1000
MAX_LOSS_POINTS = 5000


class TrainingMonitor:
    """维护一次训练运行的可观测状态，供前端周期性轮询。"""

    def __init__(self, run_id: str, total_epochs: int, mode: str = "demo") -> None:
        self._lock = threading.Lock()
        self._info = TrainingRunInfo(
            run_id=run_id,
            status="idle",
            total_epochs=total_epochs,
            mode=mode,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def set_status(self, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._info.status = status
            if error:
                self._info.error = error
            if status in ("finished", "failed", "stopped"):
                self._info.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def log(self, line: str) -> None:
        with self._lock:
            self._info.log.append(line)
            if len(self._info.log) > MAX_LOG_LINES:
                self._info.log = self._info.log[-MAX_LOG_LINES:]

    def update_epoch(self, epoch: int, loss: float, stage: int = 1) -> None:
        with self._lock:
            self._info.current_epoch = epoch
            self._info.stage = stage
            total = self._info.total_epochs or 1
            self._info.progress = round(min(100.0, epoch / total * 100.0), 2)
            self._info.loss_history.append(round(float(loss), 4))
            if len(self._info.loss_history) > MAX_LOSS_POINTS:
                self._info.loss_history = self._info.loss_history[-MAX_LOSS_POINTS:]

    def set_artifacts(self, artifacts: dict[str, str]) -> None:
        with self._lock:
            self._info.artifacts.update(artifacts)

    @property
    def info(self) -> TrainingRunInfo:
        with self._lock:
            return self._info.model_copy(deep=True)
