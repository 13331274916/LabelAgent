# LabelAgent 开发指南

本文档面向想参与开发或二次修改 LabelAgent 源码的开发者。

## 1. 环境搭建

```bash
git clone https://github.com/13331274916/LabelAgent.git
cd LabelAgent

# Python 3.10+
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt

# 启动开发服务器（修改代码后 uvicorn 需手动重启，或使用 --reload）
python run.py --port 8765
```

## 2. 代码组织速查

| 路径 | 内容 |
|---|---|
| `labelagent/config.py` | 全部常量与默认参数（新功能默认值集中在这里改） |
| `labelagent/core/` | 数据模型与 `ProjectStore` 状态中心 |
| `labelagent/annotation/` | 标注业务：导入 / Provider / 调度 / 导出 |
| `labelagent/cleaning/` | 清洗业务：质量 / 重复 / 越界 / 自动清洗 |
| `labelagent/dataset/` | 数据集业务：统计 / 划分 |
| `labelagent/environment/` | Python 环境管理 |
| `labelagent/training/` | 训练执行 / 看板 / 消融 |
| `labelagent/api/` | FastAPI 路由（每个模块一个文件） |
| `labelagent/web/` | 前端页面（原生 HTML/CSS/JS，无构建步骤） |
| `tests/` | pytest 单元测试 |

## 3. 开发约定

1. **业务逻辑不依赖 Web 框架**。新增算法请放进对应业务模块，API 路由只做参数校验与状态编排。
2. **默认参数统一放 `config.py`**，业务代码不要写死魔法数字。
3. **Pydantic 模型放 `core/models.py`**，API 响应尽量直接返回模型 dump。
4. **前端无构建步骤**：直接编辑 `labelagent/web/` 下的文件，刷新浏览器即可（首次修改后需重启服务器重新挂载静态目录）。
5. **兼容桌面版行为**：桌面版（Release v1.0.0）功能与源码版保持一致；新增功能时在 README 中同步登记。

## 4. 常见开发任务

### 4.1 新增一个标注 Provider

1. 在 `providers.py` 中继承 `BaseProvider` 实现 `annotate()` 与 `availability()`；
2. 在 `PROVIDER_REGISTRY` 中注册；
3. 在 `config.py` 的 `PROVIDERS` 列表中加入前端展示信息；
4. 如需新的默认端点，加入 `ONLINE_PROVIDER_ENDPOINTS` / `ONLINE_PROVIDER_MODELS`。

### 4.2 新增一种导出格式

1. 在 `export.py` 增加 `to_xxx()` 函数；
2. 在 `_FORMAT_EXT` 中登记扩展名；
3. `export_single()` 与 `export_zip()` 的分支中加入新格式；
4. 在 `index.html` 的导出下拉框中增加选项。

### 4.3 新增模型架构 / 损失函数 / 优化器

在 `labelagent/config.py` 的 `MODEL_ARCHS` / `LOSS_FUNCTIONS` / `OPTIMIZERS` / `SCHEDULERS` 常量中追加即可，前端选项与训练脚本自动生效。真实训练支持需要在 `scriptgen.py` 的框架检查与训练循环中接入对应模型代码。

### 4.4 接入真实训练

外部训练脚本协议：

1. 脚本通过 `sys.argv[1]` 读取 config.json；
2. 训练过程中向 stdout 输出 JSON 行 `{"type":"progress","stage":1,"epoch":n,"loss":x}`（`TrainingMonitor` 解析）；
3. 结束时在当前目录写出 `results.csv`（列：stage, epoch, train_loss, val_loss, mAP@0.5, precision）与 `best_stage1.pt` / `best_stage2.pt` / `best.pt`；
4. 返回码 0 视为成功。

## 5. 测试

```bash
pytest -v                 # 全部测试
pytest tests/test_cleaning.py -v
```

测试覆盖：五种导出格式、模糊/重复/越界检测、自动清洗、数据集划分与比例校验、训练配置校验、消融汇总与 CSV 导入。

## 6. 打包思路

桌面版（Release）使用 PyInstaller 将 `run.py` + `labelagent/` + Web 资源打包为 ONEDIR 目录：

```bash
pip install pyinstaller
pyinstaller --name LabelAgent --noconsole \
  --add-data "labelagent/web:labelagent/web" \
  --hidden-import uvicorn.logging --hidden-import uvicorn.loops.auto \
  run.py
```

> 注意：深度学习框架（torch / ultralytics）与大型模型权重不打包进 EXE，
> 训练通过「环境」模块对接外部 Python 环境完成（与桌面版 README 说明一致）。

## 7. 提交规范

- 功能开发建议同时补充单元测试；
- 修改默认参数/新增功能时同步更新 README 功能总览表；
- 提交信息使用中文或英文均可，建议描述清晰（例：`feat(annotation): 新增 XXX 导出格式`）。
