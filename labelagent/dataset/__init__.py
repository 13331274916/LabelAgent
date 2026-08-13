"""数据集模块。"""

from labelagent.dataset.split import split_dataset
from labelagent.dataset.stats import compute_stats

__all__ = ["compute_stats", "split_dataset"]
