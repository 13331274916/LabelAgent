"""清洗模块：质量诊断、重复图检测、越界检查与自动清洗。"""

from labelagent.cleaning.bounds import check_out_of_bounds, fix_out_of_bounds
from labelagent.cleaning.cleaner import auto_clean
from labelagent.cleaning.duplicate import HashCache, detect_duplicates, dhash, hamming_distance
from labelagent.cleaning.quality import diagnose, is_empty_image, laplacian_variance

__all__ = [
    "HashCache",
    "auto_clean",
    "check_out_of_bounds",
    "detect_duplicates",
    "dhash",
    "diagnose",
    "fix_out_of_bounds",
    "hamming_distance",
    "is_empty_image",
    "laplacian_variance",
]
