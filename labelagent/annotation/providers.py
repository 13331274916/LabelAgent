"""标注 Provider：本地 Grounding DINO + SAM 2 / 在线大模型 / Demo。

每个 Provider 负责将「一张图像 + 标注提示词」转换为结构化标注列表：
    annotate(image_path, prompt, label, max_objects, api_key) -> list[Annotation]

在线模型统一走 OpenAI 兼容的 chat/completions 接口并解析 JSON 输出；
本地模型需要目标机器预先准备好模型权重与依赖；
Demo Provider 基于图像内容哈希生成确定性伪标注，用于无环境体验全流程。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from labelagent.config import (
    ONLINE_PROVIDER_ENDPOINTS,
    ONLINE_PROVIDER_MODELS,
)
from labelagent.core.models import Annotation, Point

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
ANNOTATION_SCHEMA_HINT = (
    '输出要求：仅返回 JSON 数组，不要包含其他文字。'
    '数组元素为 {"label": str, "bbox": [x1, y1, x2, y2], "confidence": float}，'
    '坐标必须使用图像像素值，x1<x2、y1<y2。'
)


def _bbox_to_points(bbox: list[float], width: float, height: float) -> list[Point]:
    """将 [x1,y1,x2,y2] 像素坐标转换为多边形四点，并夹紧到图像范围。"""
    x1, y1, x2, y2 = bbox
    x1 = min(max(float(x1), 0.0), float(width))
    y1 = min(max(float(y1), 0.0), float(height))
    x2 = min(max(float(x2), 0.0), float(width))
    y2 = min(max(float(y2), 0.0), float(height))
    return [
        Point(x=x1, y=y1),
        Point(x=x2, y=y1),
        Point(x=x2, y=y2),
        Point(x=x1, y=y2),
    ]


def parse_annotation_json(text: str, width: int, height: int) -> list[dict[str, Any]]:
    """从模型返回文本中解析标注数组，兼容被 ```json ... ``` 包裹的情况。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    # 截取第一个 [ 到最后一个 ]
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox") or item.get("box") or item.get("xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        out.append(
            {
                "label": str(item.get("label") or item.get("category") or "object"),
                "bbox": bbox,
                "confidence": float(item["confidence"]) if "confidence" in item else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Provider 基类
# ---------------------------------------------------------------------------
class BaseProvider(ABC):
    id: str = ""
    name: str = ""
    mode: str = "local"          # local / online / demo
    needs_api_key: bool = False

    @abstractmethod
    def annotate(
        self,
        image_path: str,
        prompt: str,
        label: str,
        max_objects: int,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[list[Annotation], str]:
        """返回 (标注列表, 运行日志文本)。"""

    def availability(self) -> tuple[bool, str]:
        """返回 (是否可用, 说明)。默认可用。"""
        return True, "ok"


# ---------------------------------------------------------------------------
# Demo Provider（演示模式）
# ---------------------------------------------------------------------------
class DemoProvider(BaseProvider):
    """基于图像内容哈希生成确定性伪标注，方便无环境体验全流程。"""

    id = "demo"
    name = "Demo Provider（无需 API Key）"
    mode = "demo"
    needs_api_key = False

    def annotate(
        self,
        image_path: str,
        prompt: str,
        label: str,
        max_objects: int,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[list[Annotation], str]:
        try:
            with open(image_path, "rb") as f:
                digest = hashlib.md5(f.read(65536)).hexdigest()
        except OSError:
            digest = image_path

        rng = random.Random(int(digest[:8], 16))
        width = int(kwargs.get("width") or 640)
        height = int(kwargs.get("height") or 480)
        count = rng.randint(1, max(1, max_objects))

        annotations: list[Annotation] = []
        used: list[tuple[float, float, float, float]] = []
        for _ in range(count):
            for _attempt in range(20):
                w = rng.uniform(width * 0.08, width * 0.45)
                h = rng.uniform(height * 0.08, height * 0.45)
                x1 = rng.uniform(0, max(1.0, width - w))
                y1 = rng.uniform(0, max(1.0, height - h))
                x2, y2 = min(width, x1 + w), min(height, y1 + h)
                if any(
                    abs(x1 - ux) < w * 0.5 and abs(y1 - uy) < h * 0.5
                    for ux, uy, _, _ in used
                ):
                    continue
                used.append((x1, y1, x2, y2))
                annotations.append(
                    Annotation(
                        label=label,
                        points=_bbox_to_points([x1, y1, x2, y2], width, height),
                        confidence=round(rng.uniform(0.72, 0.98), 3),
                        source="demo",
                    )
                )
                break
        log = f"[demo] 生成 {len(annotations)} 个标注（label={label}）"
        return annotations, log


# ---------------------------------------------------------------------------
# 在线大模型 Provider（OpenAI 兼容接口）
# ---------------------------------------------------------------------------
class _OnlineChatProvider(BaseProvider):
    """在线 VLM 标注：将图像以 base64 发送给 chat/completions，解析 JSON 结果。"""

    id = "base_online"
    name = "Online VLM"
    mode = "online"
    needs_api_key = True
    default_endpoint = ""
    default_model = ""

    def annotate(
        self,
        image_path: str,
        prompt: str,
        label: str,
        max_objects: int,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[list[Annotation], str]:
        if not api_key:
            raise ValueError(f"Provider {self.name} 需要 API Key")

        endpoint = kwargs.get("base_url") or self.default_endpoint
        model = kwargs.get("model") or self.default_model

        import base64
        import mimetypes

        import requests

        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        user_text = (
            f"{prompt}\n请检测并标注图中的所有目标。"
            f"最多 {max_objects} 个目标。统一使用标签名：{label}。"
            f"{ANNOTATION_SCHEMA_HINT}"
        )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        width, height = kwargs.get("width", 0), kwargs.get("height", 0)
        parsed = parse_annotation_json(content, width or 640, height or 480)
        annotations: list[Annotation] = []
        for item in parsed[:max_objects]:
            lab = item["label"] if item["label"] else label
            annotations.append(
                Annotation(
                    label=lab,
                    points=_bbox_to_points(item["bbox"], width or 640, height or 480),
                    confidence=item["confidence"],
                    source=self.id,
                )
            )
        log = f"[{self.id}] 模型返回 {len(parsed)} 条，采纳 {len(annotations)} 条"
        return annotations, log

    def availability(self) -> tuple[bool, str]:
        import requests

        try:
            requests.get("https://www.baidu.com", timeout=3)  # noqa: S113 仅连通性探测
            return True, "网络可用"
        except Exception:
            return False, "当前网络不可用，在线标注将失败"


class DeepSeekVLProvider(_OnlineChatProvider):
    id = "deepseek_vl"
    name = "DeepSeek-VL"
    default_endpoint = ONLINE_PROVIDER_ENDPOINTS["deepseek_vl"]
    default_model = ONLINE_PROVIDER_MODELS["deepseek_vl"]


class Qwen2VLProvider(_OnlineChatProvider):
    id = "qwen2_vl"
    name = "阿里 Qwen2-VL"
    default_endpoint = ONLINE_PROVIDER_ENDPOINTS["qwen2_vl"]
    default_model = ONLINE_PROVIDER_MODELS["qwen2_vl"]


class OpenAIProvider(_OnlineChatProvider):
    id = "openai_gpt4o"
    name = "OpenAI GPT-4o"
    default_endpoint = ONLINE_PROVIDER_ENDPOINTS["openai_gpt4o"]
    default_model = ONLINE_PROVIDER_MODELS["openai_gpt4o"]


# ---------------------------------------------------------------------------
# 本地 Grounding DINO + SAM 2
# ---------------------------------------------------------------------------
class GroundingDinoSAM2Provider(BaseProvider):
    """本地 Grounding DINO + SAM 2 自动标注。

    说明：本项目不内置模型与权重。运行本 Provider 前需要在目标 Python 环境中安装
    groundingdino / sam2 依赖并下载权重，然后把环境路径传入 python_env。
    未安装依赖时 annotate() 会抛出带安装指引的异常。
    """

    id = "grounding_dino_sam2"
    name = "本地 Grounding DINO + SAM 2"
    mode = "local"
    needs_api_key = False

    def availability(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import groundingdino  # noqa: F401
            import sam2  # noqa: F401
            return True, "依赖已安装"
        except ImportError as e:
            return False, f"缺少依赖：{e.name}（需在训练/模型环境中安装 torch、groundingdino、sam2）"

    def annotate(
        self,
        image_path: str,
        prompt: str,
        label: str,
        max_objects: int,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[list[Annotation], str]:
        try:
            import torch  # noqa: F401
            import groundingdino  # noqa: F401
            import sam2  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"本地 Grounding DINO + SAM 2 缺少依赖 {e.name}。"
                "请安装 torch / groundingdino / sam2 并下载对应权重后重试。"
            ) from e

        # 以下为真实接入的骨架：不同版本的 GroundingDINO / SAM 2 API 差异较大，
        # 请按目标环境实际安装的版本实现。核心步骤：
        #   1. 加载 GroundingDINO 模型，以提示词 prompt 做开放词汇检测 → 边界框；
        #   2. 将检测框作为 prompt 输入 SAM 2，生成实例掩膜；
        #   3. 将掩膜轮廓转为本项目的 Annotation.points。
        raise NotImplementedError(
            "本地 GroundingDINO + SAM 2 需要在目标环境安装依赖并实现模型管线；"
            "源码中预留了集成骨架（见 providers.py:GroundingDinoSAM2Provider）。"
        )


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, BaseProvider] = {
    p.id: p
    for p in (
        GroundingDinoSAM2Provider(),
        DeepSeekVLProvider(),
        Qwen2VLProvider(),
        OpenAIProvider(),
        DemoProvider(),
    )
}


def get_provider(provider_id: str) -> BaseProvider:
    if provider_id not in PROVIDER_REGISTRY:
        raise KeyError(f"未知 Provider：{provider_id}（可选：{', '.join(PROVIDER_REGISTRY)}）")
    return PROVIDER_REGISTRY[provider_id]


def list_providers() -> list[dict[str, Any]]:
    out = []
    for pid, p in PROVIDER_REGISTRY.items():
        ok, msg = p.availability()
        out.append(
            {
                "id": p.id,
                "name": p.name,
                "mode": p.mode,
                "needs_api_key": p.needs_api_key,
                "available": ok,
                "message": msg,
            }
        )
    return out
