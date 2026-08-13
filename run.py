"""LabelAgent 启动入口。

用法：
    python run.py                         # 默认 http://127.0.0.1:8765
    python run.py --port 9000             # 自定义端口
    python run.py --demo-images 12        # 启动时生成 12 张演示图像
    python run.py --host 0.0.0.0          # 局域网访问
    python run.py --no-browser            # 不自动打开浏览器
"""

from __future__ import annotations

from labelagent.cli import main

if __name__ == "__main__":
    main()
