"""消融实验测试：结果汇总、最佳标记、CSV 导入与导出。"""

from __future__ import annotations

from labelagent.core.models import AblationGroup, AblationMetrics
from labelagent.training.ablation import AblationManager
from labelagent.training.runner import TrainingRunner


def test_ablation_manager_groups():
    mgr = AblationManager(TrainingRunner())
    g1 = mgr.add_group(AblationGroup(name="baseline", model_arch="RT-DETR"))
    g2 = mgr.add_group(AblationGroup(name="ema_on", model_arch="RT-DETR", ema=True))
    assert len(mgr.list_groups()) == 2

    try:
        mgr.add_group(AblationGroup(name="baseline", model_arch="YOLOv8-Seg"))
        assert False, "重名应报错"
    except ValueError:
        pass

    assert mgr.remove_group(g1.id) is True
    assert len(mgr.list_groups()) == 1


def test_results_best_and_improvement():
    mgr = AblationManager(TrainingRunner())
    mgr._results = [
        AblationMetrics(group_id="a", name="base", model_arch="RT-DETR", mAP50=0.500, val_loss=1.0, precision=0.6),
        AblationMetrics(group_id="b", name="ema", model_arch="RT-DETR", mAP50=0.600, val_loss=0.8, precision=0.7),
        AblationMetrics(group_id="c", name="mosaic", model_arch="YOLOv8-Seg", mAP50=0.550, val_loss=0.9, precision=0.65),
    ]
    results = mgr.results()
    best = [r for r in results if r.is_best]
    assert len(best) == 1 and best[0].name == "ema"
    by_name = {r.name: r for r in results}
    assert by_name["ema"].relative_improvement == 20.0  # (0.6-0.5)/0.5
    assert by_name["mosaic"].relative_improvement == 10.0

    csv_text = mgr.to_summary_csv()
    assert "ema" in csv_text and "20.0" in csv_text


def test_import_results_csv(tmp_path):
    import csv

    mgr = AblationManager(TrainingRunner())
    p = tmp_path / "results.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "epoch", "train_loss", "val_loss", "mAP@0.5", "precision"])
        writer.writerow([1, 1, 2.0, 1.5, 0.42, 0.5])
        writer.writerow([1, 2, 1.2, 0.9, 0.63, 0.7])

    imported = mgr.import_results_csv([str(p)], base_name="ext")
    assert len(imported) == 1
    assert imported[0].mAP50 == 0.63
    assert imported[0].note == "imported"
    assert len(mgr.results()) == 1


def test_export_summary_file(tmp_path):
    mgr = AblationManager(TrainingRunner())
    mgr._results = [AblationMetrics(group_id="a", name="base", model_arch="RT-DETR", mAP50=0.5)]
    out = mgr.export_summary(tmp_path / "ablation_summary.csv")
    assert out.endswith("ablation_summary.csv")
    assert "base" in open(out, encoding="utf-8").read()
