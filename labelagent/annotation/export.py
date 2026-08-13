"""标注格式导出：LabelMe JSON / Pascal VOC XML / YOLO TXT / MS COCO JSON / CSV，以及 ZIP 批量打包。

支持两种粒度：
- 单文件导出：export_single(image, fmt, out_dir, class_map)
- 批量打包：export_zip(images, formats, out_zip, class_map)
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Iterable, Optional

from labelagent.config import EXPORT_DIR
from labelagent.core.models import ExportResult, ImageItem


def _class_map_of(images: Iterable[ImageItem], class_map: Optional[dict[str, int]] = None) -> dict[str, int]:
    """构建类别 -> 索引映射（YOLO / COCO 使用，索引从 0 开始）。"""
    mapping = dict(class_map or {})
    for item in images:
        for ann in item.annotations:
            if ann.label and ann.label not in mapping:
                mapping[ann.label] = len(mapping)
    return mapping


# ---------------------------------------------------------------------------
# LabelMe JSON
# ---------------------------------------------------------------------------
def to_labelme_json(item: ImageItem, label_map: Optional[dict] = None) -> dict:
    """生成 LabelMe JSON（多边形/掩膜），shape_type 使用 polygon。"""
    shapes = []
    for ann in item.annotations:
        shapes.append(
            {
                "label": ann.label,
                "points": [[p.x, p.y] for p in ann.points],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        )
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": item.filename,
        "imageData": None,
        "imageHeight": item.height,
        "imageWidth": item.width,
    }


# ---------------------------------------------------------------------------
# Pascal VOC XML / LabelImg
# ---------------------------------------------------------------------------
def _fmt_number(v: float) -> str:
    """整数坐标输出为整数，否则保留一位小数。"""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


def to_voc_xml(item: ImageItem) -> str:
    """生成 Pascal VOC XML（LabelImg 兼容）。"""
    objects = []
    for ann in item.annotations:
        xmin, ymin, xmax, ymax = ann.bbox
        objects.append(
            f"""    <object>
        <name>{_xml_escape(ann.label)}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{_fmt_number(xmin)}</xmin>
            <ymin>{_fmt_number(ymin)}</ymin>
            <xmax>{_fmt_number(xmax)}</xmax>
            <ymax>{_fmt_number(ymax)}</ymax>
        </bndbox>
    </object>"""
        )
    return (
        "<annotation>\n"
        f"    <folder>LabelAgent</folder>\n"
        f"    <filename>{_xml_escape(item.filename)}</filename>\n"
        f"    <path>{_xml_escape(item.path)}</path>\n"
        "    <source>\n        <database>LabelAgent</database>\n    </source>\n"
        "    <size>\n"
        f"        <width>{item.width}</width>\n"
        f"        <height>{item.height}</height>\n"
        f"        <depth>3</depth>\n"
        "    </size>\n"
        "    <segmented>0</segmented>\n"
        + "\n".join(objects)
        + "\n</annotation>\n"
    )


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# YOLO TXT（v5/v8/v11 归一化）
# ---------------------------------------------------------------------------
def to_yolo_txt(item: ImageItem, class_map: dict[str, int]) -> str:
    """生成 YOLO TXT：每行 `class_id cx cy w h`（归一化坐标）。"""
    lines = []
    w, h = max(1, item.width), max(1, item.height)
    for ann in item.annotations:
        if ann.label not in class_map:
            continue
        xmin, ymin, xmax, ymax = ann.bbox
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{class_map[ann.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# MS COCO JSON
# ---------------------------------------------------------------------------
def to_coco_json(images: list[ImageItem], class_map: Optional[dict[str, int]] = None) -> dict:
    """生成 MS COCO JSON（annotations 使用多边形 + bbox + area）。"""
    mapping = _class_map_of(images, class_map)
    categories = [{"id": i, "name": name} for name, i in sorted(mapping.items(), key=lambda kv: kv[1])]

    coco_images, coco_anns = [], []
    ann_id = 1
    for img_id, item in enumerate(images, start=1):
        coco_images.append(
            {
                "id": img_id,
                "file_name": item.filename,
                "width": item.width,
                "height": item.height,
            }
        )
        for ann in item.annotations:
            if ann.label not in mapping:
                continue
            xmin, ymin, xmax, ymax = ann.bbox
            seg = [[p.x, p.y] for p in ann.points] or [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            bw, bh = xmax - xmin, ymax - ymin
            coco_anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": mapping[ann.label],
                    "bbox": [round(xmin, 2), round(ymin, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "segmentation": [seg],
                    "iscrowd": 0,
                    "score": ann.confidence,
                }
            )
            ann_id += 1

    return {
        "info": {
            "description": "LabelAgent exported dataset",
            "version": "1.0",
            "year": 2026,
        },
        "images": coco_images,
        "annotations": coco_anns,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# Annotation CSV
# ---------------------------------------------------------------------------
def to_annotation_csv(images: list[ImageItem]) -> str:
    """生成通用 CSV：每行一条标注。"""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["image_id", "filename", "width", "height", "label", "xmin", "ymin", "xmax", "ymax", "confidence"]
    )
    for item in images:
        for ann in item.annotations:
            xmin, ymin, xmax, ymax = ann.bbox
            writer.writerow(
                [
                    item.id,
                    item.filename,
                    item.width,
                    item.height,
                    ann.label,
                    round(xmin, 2),
                    round(ymin, 2),
                    round(xmax, 2),
                    round(ymax, 2),
                    ann.confidence if ann.confidence is not None else "",
                ]
            )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 单文件导出
# ---------------------------------------------------------------------------
_FORMAT_EXT = {
    "labelme": ".json",
    "voc": ".xml",
    "yolo": ".txt",
    "coco": ".json",
    "csv": ".csv",
}


def export_single(
    item: ImageItem,
    fmt: str,
    out_dir: str | Path = EXPORT_DIR,
    class_map: Optional[dict[str, int]] = None,
) -> str:
    """将单张图像的标注导出为指定格式，返回生成文件路径。

    fmt 取值：labelme / voc / yolo / coco / csv
    """
    fmt = fmt.lower()
    if fmt not in _FORMAT_EXT:
        raise ValueError(f"不支持的导出格式：{fmt}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(item.filename).stem
    if fmt == "labelme":
        data = to_labelme_json(item)
        text = json.dumps(data, ensure_ascii=False, indent=2)
    elif fmt == "voc":
        text = to_voc_xml(item)
    elif fmt == "yolo":
        mapping = _class_map_of([item], class_map)
        text = to_yolo_txt(item, mapping)
    elif fmt == "coco":
        data = to_coco_json([item], class_map)
        text = json.dumps(data, ensure_ascii=False, indent=2)
    else:  # csv
        text = to_annotation_csv([item])

    target = out_dir / f"{stem}{_FORMAT_EXT[fmt]}"
    target.write_text(text, encoding="utf-8")
    return str(target)


# ---------------------------------------------------------------------------
# ZIP 批量导出
# ---------------------------------------------------------------------------
def export_zip(
    images: list[ImageItem],
    formats: Iterable[str],
    out_zip: str | Path = EXPORT_DIR / "labelme_dataset_project.zip",
    class_map: Optional[dict[str, int]] = None,
) -> str:
    """将一批图像的指定格式标注打包为 ZIP。

    默认输出文件名 labelme_dataset_project.zip（与桌面版前端一致）。
    """
    formats = [f.lower() for f in formats if f.lower() in _FORMAT_EXT]
    if not formats:
        raise ValueError("没有可导出的格式")

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    mapping = _class_map_of(images, class_map)
    files_written = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in images:
            stem = Path(item.filename).stem
            for fmt in formats:
                if fmt == "labelme":
                    text = json.dumps(to_labelme_json(item), ensure_ascii=False, indent=2)
                elif fmt == "voc":
                    text = to_voc_xml(item)
                elif fmt == "yolo":
                    text = to_yolo_txt(item, mapping)
                elif fmt == "coco":
                    text = json.dumps(to_coco_json([item], mapping), ensure_ascii=False, indent=2)
                else:  # csv
                    text = to_annotation_csv([item])
                # 按格式分子目录存放，避免 labelme/coco 同为 .json 时文件重名
                zf.writestr(
                    f"{fmt}/{stem}{_FORMAT_EXT[fmt]}",
                    text,
                    compress_type=zipfile.ZIP_DEFLATED,
                )
                files_written += 1
        # COCO 全量文件单独输出一份（含全部 images 的 annotation）
        if "coco" in formats and len(images) > 1:
            zf.writestr(
                "coco/coco_annotations.json",
                json.dumps(to_coco_json(images, mapping), ensure_ascii=False, indent=2),
            )
            files_written += 1
        if "csv" in formats and len(images) > 1:
            zf.writestr("csv/annotations.csv", to_annotation_csv(images))
            files_written += 1

    return str(out_zip)


def export_all_formats(
    images: list[ImageItem],
    out_dir: str | Path = EXPORT_DIR,
    class_map: Optional[dict[str, int]] = None,
) -> ExportResult:
    """将全部图像以五种格式分别导出到 out_dir 子目录。"""
    out_dir = Path(out_dir)
    mapping = _class_map_of(images, class_map)
    files: list[str] = []
    for fmt in _FORMAT_EXT:
        d = out_dir / fmt
        for item in images:
            files.append(export_single(item, fmt, d, mapping))
    return ExportResult(format="all", count=len(images), files=files)
