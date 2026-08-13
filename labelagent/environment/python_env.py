"""Python 环境模块：解释器检测、手动导入、依赖包查看。

说明：LabelAgent 自身（GUI/Web/图像处理）所需依赖已内置；
深度学习训练（RT-DETR / YOLO / Swin 等）需要外部 Python 环境，
本模块负责发现并导入该环境，供训练执行器调用。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from labelagent.core.models import PackageInfo, PythonEnvInfo

# 训练关键依赖（用于 Status 标记）
REQUIRED_PACKAGES = ["torch", "ultralytics", "opencv-python", "transformers", "numpy"]


def _run_python(python_exe: str, args: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [python_exe, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


def detect_python_envs() -> list[PythonEnvInfo]:
    """发现可用的 Python 解释器列表。"""
    found: dict[str, PythonEnvInfo] = {}

    # 当前进程解释器
    current = sys.executable
    if current:
        found[current] = _inspect(current)

    # PATH 中的 python / python3
    for name in ("python", "python3"):
        p = shutil.which(name)
        if p and p not in found:
            found[p] = _inspect(p)

    # 常见 conda / 虚拟环境路径（跨平台）
    candidates = []
    home = Path.home()
    candidates += sorted(home.glob("miniconda3/envs/*/python.exe")) if sys.platform == "win32" else sorted(home.glob("miniconda3/envs/*/bin/python*"))
    candidates += sorted(home.glob("anaconda3/envs/*/python.exe")) if sys.platform == "win32" else sorted(home.glob("anaconda3/envs/*/bin/python*"))
    if sys.platform == "win32":
        for drive in ("C:/", "D:/", "E:/"):
            for env_root in ("Miniconda3", "Anaconda3", "miniconda3", "anaconda3"):
                candidates += sorted(Path(drive).glob(f"{env_root}/envs/*/python.exe"))
    for p in candidates:
        if p not in found:
            found[str(p)] = _inspect(str(p))

    return list(found.values())


def import_python_env(python_exe: str) -> PythonEnvInfo:
    """手动导入外部 Python 环境并读取版本与依赖包。"""
    path = str(Path(python_exe).resolve())
    if not Path(path).exists():
        return PythonEnvInfo(path=path, valid=False, error="文件不存在")
    info = _inspect(path)
    return info


def _inspect(python_exe: str) -> PythonEnvInfo:
    info = PythonEnvInfo(path=python_exe)
    code = (
        "import sys;"
        "print(sys.version.split()[0]);"
        "print(sys.prefix)"
    )
    rc, out, err = _run_python(python_exe, ["-c", code])
    if rc != 0:
        info.valid = False
        info.error = (err or out).strip()[:300]
        return info
    lines = out.strip().splitlines()
    if len(lines) >= 2:
        info.version = lines[0]
        info.prefix = lines[1]
    info.packages = _list_packages(python_exe)
    return info


def _list_packages(python_exe: str) -> list[PackageInfo]:
    code = (
        "import json, importlib.metadata as md;"
        "print(json.dumps([(d.metadata['Name'], d.version) for d in md.distributions()]))"
    )
    rc, out, err = _run_python(python_exe, ["-c", code])
    if rc != 0:
        # 回退到 pip list --format=json
        rc, out, err = _run_python(python_exe, ["-m", "pip", "list", "--format=json"])
        if rc != 0:
            return []
        try:
            data = json.loads(out)
            pkgs = [(d["name"], d["version"]) for d in data]
        except (json.JSONDecodeError, KeyError):
            return []
    else:
        try:
            pkgs = json.loads(out)
        except json.JSONDecodeError:
            return []
    return [
        PackageInfo(name=name, version=version, required=name.lower() in {r.lower() for r in REQUIRED_PACKAGES})
        for name, version in pkgs
    ]
