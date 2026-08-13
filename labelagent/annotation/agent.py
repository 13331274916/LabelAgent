"""Agent 自动标注调度：批量对项目内图像执行所选 Provider 的标注。"""

from __future__ import annotations

import threading
import traceback
from typing import Callable, Optional

from labelagent.annotation.providers import get_provider
from labelagent.core.models import AgentTaskStatus, ImageItem
from labelagent.core.store import ProjectStore

ProgressCallback = Callable[[int, int, str], None]


class AgentAnnotator:
    """在后台线程中批量执行 Agent 自动标注，并暴露任务状态。"""

    def __init__(self, store: ProjectStore) -> None:
        self._store = store
        self._status = AgentTaskStatus()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def status(self) -> AgentTaskStatus:
        with self._lock:
            return self._status.model_copy(deep=True)

    def _log(self, line: str) -> None:
        with self._lock:
            self._status.log.append(line)
            self._status.log = self._status.log[-500:]

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        provider_id: str,
        prompt: str,
        label: str,
        max_objects: int,
        api_key: Optional[str] = None,
        image_ids: Optional[list[str]] = None,
    ) -> AgentTaskStatus:
        """启动一次标注任务（非阻塞）。"""
        if self.is_running():
            raise RuntimeError("已有标注任务正在运行")
        with self._lock:
            self._status = AgentTaskStatus(
                running=True,
                provider=provider_id,
                total=0,
                done=0,
                log=[f"开始 Agent 自动标注：provider={provider_id}"],
            )
        self._thread = threading.Thread(
            target=self._run,
            kwargs={
                "provider_id": provider_id,
                "prompt": prompt,
                "label": label,
                "max_objects": max_objects,
                "api_key": api_key,
                "image_ids": image_ids,
            },
            daemon=True,
        )
        self._thread.start()
        return self.status

    def _run(
        self,
        provider_id: str,
        prompt: str,
        label: str,
        max_objects: int,
        api_key: Optional[str],
        image_ids: Optional[list[str]],
    ) -> None:
        provider = get_provider(provider_id)
        images: list[ImageItem] = [
            it for it in self._store.list_images() if (image_ids is None or it.id in image_ids)
        ]
        with self._lock:
            self._status.total = len(images)
        ok = 0
        try:
            for idx, item in enumerate(images, start=1):
                with self._lock:
                    self._status.current_image = item.filename
                try:
                    annotations, log = provider.annotate(
                        item.path,
                        prompt=prompt,
                        label=label,
                        max_objects=max_objects,
                        api_key=api_key,
                        width=item.width,
                        height=item.height,
                    )
                    self._store.set_annotations(item.id, annotations)
                    ok += 1
                    self._log(f"[{idx}/{len(images)}] {item.filename} → {len(annotations)} 个标注（{log}）")
                except Exception as e:  # noqa: BLE001 单张失败不中断整个任务
                    self._log(f"[{idx}/{len(images)}] {item.filename} 标注失败：{e}")
                with self._lock:
                    self._status.done = idx
            self._log(f"任务完成：成功 {ok}/{len(images)} 张")
            with self._lock:
                self._status.finished = True
        except Exception as e:  # noqa: BLE001
            self._log(f"任务异常终止：{e}")
            tb = traceback.format_exc()
            self._log(tb[-800:])
            with self._lock:
                self._status.error = str(e)
                self._status.finished = True
        finally:
            with self._lock:
                self._status.running = False
