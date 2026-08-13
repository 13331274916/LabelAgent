"""数据集模块测试：类别统计与 Train/Val 划分。"""

from __future__ import annotations

import yaml

from tests.conftest import make_annotation, make_image, make_item


def test_compute_stats(tmp_project):
    store, _ = tmp_project
    from labelagent.dataset.stats import compute_stats

    stats = compute_stats(store)
    assert stats.image_count == 2
    assert stats.annotation_count == 2
    assert stats.per_class == {"scratch_defect": 1, "crack": 1}


def test_split_dataset(tmp_project):
    store, tmp_path = tmp_project
    from labelagent.dataset.split import split_dataset

    result = split_dataset(store, train_ratio=0.50, dataset_name="test_ds", output_dir=tmp_path / "ds")
    assert result.train_count == 1
    assert result.val_count == 1
    assert result.train_ratio == 0.5
    assert result.dataset_name == "test_ds"

    yaml_path = tmp_path / "ds" / "dataset.yaml"
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["nc"] == 2

    # 图像与标注文件应已生成
    assert (tmp_path / "ds" / "images" / "train").exists()
    labels = list((tmp_path / "ds" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 1


def test_split_ratio_validation(tmp_project):
    store, tmp_path = tmp_project
    from labelagent.dataset.split import split_dataset

    try:
        split_dataset(store, train_ratio=0.10, output_dir=tmp_path / "bad")
        assert False, "比例过小应报错"
    except ValueError:
        pass

    # 0.80 比例应落在合法范围
    result = split_dataset(store, train_ratio=0.80, output_dir=tmp_path / "ok")
    assert result.train_ratio == 0.8
