"""训练与消融模块。"""

from labelagent.training.ablation import AblationManager
from labelagent.training.config import options
from labelagent.training.monitor import TrainingMonitor
from labelagent.training.runner import TrainingRunner
from labelagent.training.scriptgen import dump_config, generate_train_script

__all__ = [
    "AblationManager",
    "TrainingMonitor",
    "TrainingRunner",
    "dump_config",
    "generate_train_script",
    "options",
]
