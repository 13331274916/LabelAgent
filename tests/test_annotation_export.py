"""标注导出格式测试：LabelMe / VOC / YOLO / COCO / CSV。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tests.conftest import make_annotation, make_image, make_item


def test_labelme_json(tmp_path):
    img = make_image(tmp_path / "l.png")
    item = make_item(str(img), annotations=[make_annotation()])
    from labelagent.annotation.export import to_labelme_json

    data = to_labelme_json(item)
    assert data["imageWidth"] == 320
    assert data["imageHeight"] == 240
    assert len(data["shapes"]) == 1
    shape = data["shapes"][0]
    assert shape["label"] == "scratch_defect"
    assert shape["shape_type"] == "polygon"
    assert len(shape["points"]) == 4


def test_voc_xml(tmp_path):
    img = make_image(tmp_path / "v.png")
    item = make_item(str(img), annotations=[make_annotation(label="crack")])
    from labelagent.annotation.export import to_voc_xml

    xml = to_voc_xml(item)
    assert "<annotation>" in xml
    assert "<name>crack</name>" in xml
    assert "<xmin>10</xmin>" in xml
    assert "<xmax>60</xmax>" in xml


def test_yolo_txt(tmp_path):
    img = make_image(tmp_path / "y.png")
    item = make_item(str(img), annotations=[make_annotation()])
    from labelagent.annotation.export import to_yolo_txt

    txt = to_yolo_txt(item, {"scratch_defect": 0}).strip()
    # class_id cx cy w h，全部归一化在 0~1 之间
    parts = [float(x) for x in txt.split()]
    assert parts[0] == 0
    assert all(0.0 <= v <= 1.0 for v in parts[1:])


def test_coco_json(tmp_path):
    img_a = make_image(tmp_path / "a.png")
    img_b = make_image(tmp_path / "b.png", color="lightblue")
    items = [
        make_item(str(img_a), annotations=[make_annotation("scratch_defect", (10, 10, 60, 60))]),
        make_item(str(img_b), annotations=[make_annotation("crack", (30, 30, 90, 70))]),
    ]
    from labelagent.annotation.export import to_coco_json

    coco = to_coco_json(items)
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 2
    assert {c["name"] for c in coco["categories"]} == {"scratch_defect", "crack"}
    ann = coco["annotations"][0]
    assert ann["bbox"] == [10.0, 10.0, 50.0, 50.0]
    assert ann["area"] == 2500.0


def test_annotation_csv(tmp_path):
    img = make_image(tmp_path / "c.png")
    item = make_item(str(img), annotations=[make_annotation()])
    from labelagent.annotation.export import to_annotation_csv

    csv_text = to_annotation_csv([item])
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("image_id,filename")
    assert item.filename in csv_text
    assert "scratch_defect" in csv_text


def test_export_single_and_zip(tmp_path):
    img_a = make_image(tmp_path / "a.png")
    img_b = make_image(tmp_path / "b.png", color="lightblue")
    items = [
        make_item(str(img_a), annotations=[make_annotation("scratch_defect")]),
        make_item(str(img_b), annotations=[make_annotation("crack", (5, 5, 40, 40))]),
    ]
    from labelagent.annotation.export import export_single, export_zip

    out_dir = tmp_path / "out"
    for fmt in ("labelme", "voc", "yolo", "coco", "csv"):
        path = export_single(items[0], fmt, out_dir=out_dir)
        assert Path(path).exists()

    zip_path = export_zip(items, ["labelme", "yolo", "coco"], out_zip=tmp_path / "ds.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "labelme/a.json" in names and "yolo/a.txt" in names
        assert "labelme/b.json" in names and "yolo/b.txt" in names
        assert "coco/coco_annotations.json" in names
        # 同一图像不同格式不应重名
        assert len(names) == len(set(names))
