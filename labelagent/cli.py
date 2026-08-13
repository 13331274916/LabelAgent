"""命令行入口：`labelagent` 命令（等价于 python run.py）。"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="labelagent", description="LabelAgent - CV 数据标注与训练实验平台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--demo-images", type=int, default=0, help="启动时自动生成的演示图像数量")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from labelagent.api.app import create_app

    app = create_app()
    if args.demo_images > 0:
        from labelagent.demo import generate_demo_images

        items = generate_demo_images(app.state.la.store, count=args.demo_images)
        print(f"已生成 {len(items)} 张演示图像并导入项目")

    if not args.no_browser:
        import threading
        import webbrowser

        def _open() -> None:
            import time

            time.sleep(1.2)
            webbrowser.open(f"http://{args.host}:{args.port}")

        threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
