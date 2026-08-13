"""全局配置与默认参数。

集中管理 LabelAgent 的路径、默认值与常量，便于后续将配置迁移到配置文件/环境变量。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 项目根目录（labelagent/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 运行期工作目录：日志、缓存、导出、演示图像、训练产物
WORKSPACE_DIR = Path(os.environ.get("LABELAGENT_WORKSPACE", PROJECT_ROOT / "workspace"))
EXPORT_DIR = WORKSPACE_DIR / "exports"
CACHE_DIR = WORKSPACE_DIR / "cache"
RUNS_DIR = WORKSPACE_DIR / "runs"
DEMO_IMAGE_DIR = WORKSPACE_DIR / "demo_images"
LOG_FILE = WORKSPACE_DIR / "labelagent.log"

WEB_DIR = Path(__file__).resolve().parent / "web"

# ---------------------------------------------------------------------------
# 图像
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# 标注模块默认值（与桌面版 README 对齐）
# ---------------------------------------------------------------------------
DEFAULT_LABEL = "scratch_defect"      # 统一标签名称
DEFAULT_MAX_OBJECTS = 5               # 单图目标数量上限
DEFAULT_PROMPT = "标注汽车, 划痕, 缺陷"  # 文本标注提示词

PROVIDERS = [
    {
        "id": "grounding_dino_sam2",
        "name": "本地 Grounding DINO + SAM 2",
        "mode": "local",
        "needs_api_key": False,
    },
    {
        "id": "deepseek_vl",
        "name": "DeepSeek-VL",
        "mode": "online",
        "needs_api_key": True,
    },
    {
        "id": "qwen2_vl",
        "name": "阿里 Qwen2-VL",
        "mode": "online",
        "needs_api_key": True,
    },
    {
        "id": "openai_gpt4o",
        "name": "OpenAI GPT-4o",
        "mode": "online",
        "needs_api_key": True,
    },
    {
        "id": "demo",
        "name": "Demo Provider（无需 API Key）",
        "mode": "demo",
        "needs_api_key": False,
    },
]

# 在线模型默认服务地址（可被 API Key 表单中的 base_url 覆盖）
ONLINE_PROVIDER_ENDPOINTS = {
    "deepseek_vl": "https://api.deepseek.com/chat/completions",
    "qwen2_vl": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "openai_gpt4o": "https://api.openai.com/v1/chat/completions",
}

ONLINE_PROVIDER_MODELS = {
    "deepseek_vl": "deepseek-vl2",
    "qwen2_vl": "qwen2.5-vl-72b-instruct",
    "openai_gpt4o": "gpt-4o",
}

# ---------------------------------------------------------------------------
# 清洗模块默认值
# ---------------------------------------------------------------------------
DEFAULT_BLUR_THRESHOLD = 100.0    # 模糊度阈值（Laplacian 方差），对应桌面版 cleanBlurThresh
HASH_CACHE_FILE = CACHE_DIR / "image_hashes.json"
LSH_BUCKETS = 8                   # LSH 分桶数（按位切分）

# ---------------------------------------------------------------------------
# 数据集模块默认值
# ---------------------------------------------------------------------------
DEFAULT_DATASET_NAME = "defect_dataset_split_v1"
SPLIT_RATIO_MIN = 0.50
SPLIT_RATIO_MAX = 0.95
SPLIT_RATIO_STEP = 0.05
DEFAULT_SPLIT_RATIO = 0.80
SPLIT_SEED = 2026

# ---------------------------------------------------------------------------
# 训练模块默认值（与桌面版 README 对齐）
# ---------------------------------------------------------------------------
DEFAULT_TRAINING = {
    "epochs": 12,
    "batch": 8,
    "img_size": 640,
    "lr": 0.01,
    "two_stage": True,
    "stage1_epochs": 8,
    "stage2_epochs": 4,
}

MODEL_ARCHS = [
    "RT-DETR",
    "Deformable-DETR",
    "Swin Transformer Seg",
    "YOLOv8-Seg",
    "YOLOv11",
    "ResNet50-FPN (PyTorch)",
]

LOSS_FUNCTIONS = [
    "CIoU Loss + BCE",
    "Focal Loss + EIoU",
    "SIoU Loss",
    "DIoU Loss",
    "GIoU Loss",
]

OPTIMIZERS = ["AdamW", "SGD + Momentum", "Lion", "Adam", "RMSprop"]

SCHEDULERS = [
    "Cosine Annealing",
    "Polynomial Decay",
    "Step LR Decay",
    "Exponential Decay",
    "Linear Decay",
]

# 训练提分策略（开关项）
STRATEGIES = {
    "warmup": "LR Warmup",
    "ema": "EMA 权重移动平均",
    "amp": "AMP 自动混合精度",
    "cosine_adaptive_decay": "余弦退火自适应衰减",
    "mosaic": "Mosaic 四图混合",
    "label_smoothing": "标签平滑",
}

ABLATION_VARIANTS = ["RT-DETR", "YOLOv8-Seg", "Deformable-DETR", "ResNet50"]


def ensure_workspace() -> None:
    """创建运行期工作目录。"""
    for d in (WORKSPACE_DIR, EXPORT_DIR, CACHE_DIR, RUNS_DIR, DEMO_IMAGE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    """初始化日志（控制台 + workspace/labelagent.log）。"""
    ensure_workspace()
    logger = logging.getLogger("labelagent")
    if logger.handlers:  # 防止重复初始化
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
