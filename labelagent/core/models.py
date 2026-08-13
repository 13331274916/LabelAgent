"""LabelAgent 数据模型。

使用 Pydantic 定义核心数据结构：标注（Annotation）、图像（ImageItem）、
清洗/数据集/训练/消融相关的各类结果对象。这些模型同时用于：
1. 业务层（annotation / cleaning / dataset / training）内部流转；
2. API 层的请求/响应序列化。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from labelagent.config import DEFAULT_DATASET_NAME, DEFAULT_LABEL, STRATEGIES


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# 标注
# ---------------------------------------------------------------------------
class Point(BaseModel):
    """多边形顶点（像素坐标）。"""

    x: float
    y: float


class Annotation(BaseModel):
    """一条标注：多边形/掩膜轮廓点 + 标签 + 置信度。"""

    id: str = Field(default_factory=_new_id)
    label: str = Field(default=DEFAULT_LABEL)
    points: list[Point] = Field(default_factory=list)
    confidence: Optional[float] = None
    source: str = "agent"  # agent / manual / imported

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """返回 (xmin, ymin, xmax, ymax)。无顶点时返回 (0, 0, 0, 0)。"""
        if not self.points:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center(self) -> tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bbox
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return self.bbox

    def clamp(self, width: float, height: float) -> "Annotation":
        """将标注裁剪到图像范围内（越界纠偏）。"""
        pts = [
            Point(
                x=min(max(p.x, 0.0), float(width)),
                y=min(max(p.y, 0.0), float(height)),
            )
            for p in self.points
        ]
        return self.model_copy(update={"points": pts})

    def model_dump_safe(self) -> dict[str, Any]:
        return self.model_dump()


class ImageItem(BaseModel):
    """一张已导入的图像及其标注。"""

    id: str = Field(default_factory=_new_id)
    path: str
    filename: str
    width: int = 0
    height: int = 0
    annotations: list[Annotation] = Field(default_factory=list)
    # 清洗诊断标记
    is_blurry: bool = False
    duplicate_of: Optional[str] = None   # 重复图：保留的第一张图 id
    has_oob: bool = False                 # 是否存在越界标注
    is_empty: bool = False                # 是否空图（无有效内容）
    added_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def annotation_count(self) -> int:
        return len(self.annotations)

    def stat(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "annotation_count": self.annotation_count,
            "is_blurry": self.is_blurry,
            "duplicate_of": self.duplicate_of,
            "has_oob": self.has_oob,
            "is_empty": self.is_empty,
        }


# ---------------------------------------------------------------------------
# 标注模块
# ---------------------------------------------------------------------------
class AgentTaskStatus(BaseModel):
    """一次 Agent 自动标注任务的状态。"""

    running: bool = False
    provider: str = ""
    total: int = 0
    done: int = 0
    current_image: str = ""
    log: list[str] = Field(default_factory=list)
    finished: bool = False
    error: Optional[str] = None


class ExportResult(BaseModel):
    """导出结果。"""

    format: str
    count: int = 0
    files: list[str] = Field(default_factory=list)
    zip_path: Optional[str] = None


# ---------------------------------------------------------------------------
# 清洗模块
# ---------------------------------------------------------------------------
class DiagnosticResult(BaseModel):
    """质量诊断结果。"""

    health_score: float = 100.0
    scanned: int = 0
    anomalies: int = 0
    elapsed_ms: float = 0.0
    blurry_count: int = 0
    duplicate_count: int = 0
    oob_count: int = 0
    empty_count: int = 0
    details: dict[str, Any] = Field(default_factory=dict)

    def summarize(self) -> str:
        return (
            f"健康得分 {self.health_score:.0f}/100，扫描 {self.scanned} 张，"
            f"异常 {self.anomalies} 项（模糊 {self.blurry_count} / "
            f"重复 {self.duplicate_count} / 越界 {self.oob_count} / 空图 {self.empty_count}）"
        )


class CleanResult(BaseModel):
    """自动清洗结果。"""

    kept: int = 0
    removed: int = 0
    fixed_oob: int = 0
    removed_ids: list[str] = Field(default_factory=list)
    fixed_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 数据集模块
# ---------------------------------------------------------------------------
class DatasetStats(BaseModel):
    """类别统计结果。"""

    image_count: int = 0
    annotation_count: int = 0
    per_class: dict[str, int] = Field(default_factory=dict)


class SplitResult(BaseModel):
    """Train/Val 划分结果。"""

    dataset_name: str
    train_count: int = 0
    val_count: int = 0
    train_ratio: float = 0.0
    yaml_path: Optional[str] = None
    output_dir: Optional[str] = None
    classes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 环境模块
# ---------------------------------------------------------------------------
class PackageInfo(BaseModel):
    name: str
    version: str
    required: bool = False


class PythonEnvInfo(BaseModel):
    path: str
    version: str = ""
    prefix: str = ""
    packages: list[PackageInfo] = Field(default_factory=list)
    valid: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# 训练模块
# ---------------------------------------------------------------------------
class TrainingConfig(BaseModel):
    """一次训练任务的全部配置。"""

    # 基础参数（与桌面版默认值对齐）
    epochs: int = Field(default=12, ge=1, le=1000)
    batch: int = Field(default=8, ge=1, le=1024)
    img_size: int = Field(default=640, ge=32, le=4096)
    lr: float = Field(default=0.01, gt=0.0, lt=1.0)

    # 架构与算法选择
    model_arch: str = "RT-DETR"
    loss: str = "CIoU Loss + BCE"
    optimizer: str = "AdamW"
    scheduler: str = "Cosine Annealing"

    # 两阶段训练
    two_stage: bool = True
    stage1_epochs: int = Field(default=8, ge=1)
    stage2_epochs: int = Field(default=4, ge=1)

    # 训练提分策略开关
    strategies: dict[str, bool] = Field(default_factory=dict)

    # 数据集与环境
    dataset_name: str = Field(default=DEFAULT_DATASET_NAME)
    dataset_dir: Optional[str] = None
    python_exe: Optional[str] = None

    # 训练目标（从当前项目数据自动统计）
    classes: list[str] = Field(default_factory=list)

    # 消融实验使用的变体标识
    variant_name: Optional[str] = None

    @field_validator("stage1_epochs")
    @classmethod
    def _stage1_lt_epochs(cls, v: int, info) -> int:
        total = info.data.get("epochs", 12)
        if v >= total:
            raise ValueError(f"Stage 1 epochs ({v}) 必须小于总 Epochs ({total})")
        return v

    @field_validator("stage2_epochs")
    @classmethod
    def _stage2_lt_epochs(cls, v: int, info) -> int:
        total = info.data.get("epochs", 12)
        if v >= total:
            raise ValueError(f"Stage 2 epochs ({v}) 必须小于总 Epochs ({total})")
        return v

    @property
    def effective_strategies(self) -> dict[str, str]:
        """返回已开启的策略（英文键 -> 中文名）。"""
        return {k: STRATEGIES[k] for k, v in self.strategies.items() if v and k in STRATEGIES}


class AblationGroup(BaseModel):
    """一组消融实验配置。"""

    id: str = Field(default_factory=_new_id)
    name: str = "experiment"
    model_arch: str = "RT-DETR"
    warmup: bool = False
    ema: bool = False
    mosaic: bool = False
    loss: str = "CIoU Loss + BCE"
    optimizer: str = "AdamW"
    scheduler: str = "Cosine Annealing"
    epochs: int = 12
    batch: int = 8
    lr: float = 0.01


class AblationMetrics(BaseModel):
    """一组消融实验的评估指标。"""

    group_id: str
    name: str = ""
    model_arch: str = ""
    val_loss: float = 0.0
    mAP50: float = 0.0
    precision: float = 0.0
    relative_improvement: Optional[float] = None
    is_best: bool = False
    note: str = ""


class TrainingRunInfo(BaseModel):
    """训练运行记录（含看板状态）。"""

    run_id: str = Field(default_factory=_new_id)
    status: Literal["idle", "running", "finished", "failed", "stopped"] = "idle"
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    stage: int = 0
    loss_history: list[float] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    mode: str = "demo"  # demo / external
    artifacts: dict[str, str] = Field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
