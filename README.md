# LabelAgent

> 面向计算机视觉的数据智能标注、数据清洗、数据集构建与训练实验桌面平台。

LabelAgent 是一套面向计算机视觉（CV）数据处理与实验管理的平台：支持 **AI Agent 自动图像标注**、**数据质量诊断与清洗**、**YOLO/COCO 等数据集自动构建**、**Train/Val 划分**，以及**模型训练参数配置、两阶段训练、多组消融实验与结果可视化**。

- 🖥️ **桌面版**：Windows 10 / 11 一键运行，无需配置完整开发环境（前往 [Releases](https://github.com/13331274916/LabelAgent/releases) 下载 v1.0.0）
- 💻 **源码版**：本仓库提供完整 Python 源码，可在 Windows / Linux / macOS 上以本地服务 + 浏览器方式运行，便于二次开发与学习

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

---

## 目录

1. [核心能力](#核心能力)
2. [功能总览](#功能总览)
3. [快速开始](#快速开始)
4. [目录结构](#目录结构)
5. [源码架构](#源码架构)
6. [接口说明](#接口说明)
7. [文档索引](#文档索引)
8. [已知事项与后续规划](#已知事项与后续规划)
9. [许可证](#许可证)

---

## 核心能力

| 模块 | 能力 |
|---|---|
| 🏷️ **标注** | 文件夹/多图导入、Agent 自动标注、标注浏览与修订、五格式导出（LabelMe / VOC / YOLO / COCO / CSV）、ZIP 批量打包 |
| 🧹 **清洗** | 模糊图检测、重复图检测（LSH / Pairwise）、越界标注检查与纠偏、空图清理、哈希缓存、健康得分 |
| 📂 **数据集** | 类别统计、Train / Val 自动划分、自定义数据集名称、自动生成 `dataset.yaml` |
| 🐍 **环境** | Python 解释器检测与手动导入、依赖包与版本查看 |
| 🚀 **训练与消融** | 模型 / 损失函数 / 优化器 / 学习率调度 / 训练策略配置、两阶段训练、实时训练看板、多组消融实验、CSV 导入对比、结果导出 |

### Agent 自动标注

- **本地模型**：Grounding DINO + SAM 2（需要本机准备模型权重与依赖）
- **在线模型**：DeepSeek-VL、阿里 Qwen2-VL、OpenAI GPT-4o（需要 API Key）
- **演示模式**：内置 Demo Provider，无需 API Key 即可完整体验标注 → 清洗 → 数据集 → 训练的全流程

### 训练与消融

- 模型架构：RT-DETR、Deformable-DETR、Swin Transformer Seg、YOLOv8-Seg / YOLOv11、ResNet50-FPN
- 损失函数：CIoU + BCE、Focal + EIoU、SIoU、DIoU、GIoU
- 优化器：AdamW、SGD + Momentum、Lion、Adam、RMSprop
- 调度策略：Cosine Annealing、Polynomial Decay、Step LR Decay、Exponential Decay、Linear Decay
- 训练策略：LR Warmup、EMA、AMP、余弦退火自适应衰减、Mosaic、标签平滑
- 两阶段训练：Stage 1 / Stage 2，产出 `best_stage1.pt`、`best_stage2.pt`、`best.pt`
- 消融实验：多变体配置、批量运行、Val Loss / mAP@0.5 对比、`ablation_summary.csv` 导出、对比图

---

## 功能总览

| 模块 | 功能 |
|---|---|
| 标注 | 文件夹/多图导入、Agent 自动标注、模型选择、API Key、标签规则、图库浏览 |
| 标注修订 | 标签重命名、删除错误标注、标注可视化 |
| 导出 | LabelMe、VOC/LabelImg、YOLO、COCO、CSV、ZIP |
| 清洗 | 模糊检测、重复图检测、越界检查、空图清理、自动清洗 |
| 加速 | 哈希缓存、LSH、Pairwise 精确匹配、清空缓存 |
| 数据集 | 类别统计、Train/Val 比例、自定义名称、dataset.yaml |
| 环境 | Python 解释器检测、手动导入、依赖包和版本查看 |
| 模型 | RT-DETR、Deformable-DETR、Swin Transformer Seg、YOLOv8-Seg/YOLOv11、ResNet50-FPN |
| Loss | CIoU+BCE、Focal+EIoU、SIoU、DIoU、GIoU |
| Optimizer | AdamW、SGD、Lion、Adam、RMSprop |
| Scheduler | Cosine、Polynomial、Step、Exponential、Linear |
| 训练策略 | Warmup、EMA、AMP、Cosine、Mosaic、Label Smoothing |
| 两阶段训练 | Stage 1 / Stage 2、best_stage1.pt、best_stage2.pt、best.pt |
| 训练监控 | Epoch、百分比、进度条、Loss 图、控制台日志 |
| 训练导出 | results.csv、Stage 1/2 权重、best.pt |
| 消融实验 | 多变体配置、模型/策略对比、批量运行 |
| 外部实验 | 批量导入多个 results.csv |
| 消融指标 | Val Loss、mAP@0.5、Precision、最佳模型、相对提升 |
| 消融导出 | ablation_summary.csv、实验对比图 |

---

## 快速开始

### 方式一：下载 Windows 桌面版（推荐给普通用户）

1. 前往 [Releases](https://github.com/13331274916/LabelAgent/releases) 下载最新版本（v1.0.0）；
2. 解压完整压缩包，**保留 `_internal` 目录**（程序运行依赖，不能删除或分离）；
3. 双击 `LabelAgent.exe` 启动。

> 详细的桌面版使用手册见 [README_labelagent.md](./README_labelagent.md)。

### 方式二：从源码运行

```bash
# 1. 克隆仓库
git clone https://github.com/13331274916/LabelAgent.git
cd LabelAgent

# 2. 创建虚拟环境（Python 3.10+）
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动（默认 http://127.0.0.1:8765）
python run.py

# 可选参数
python run.py --host 0.0.0.0 --port 8765   # 局域网访问
python run.py --demo-images 12             # 启动时自动生成演示图像并导入
python run.py --no-browser                 # 不自动打开浏览器
```

启动后浏览器会打开 LabelAgent 主界面，包含五个功能模块（标注 / 清洗 / 数据集 / 环境 / 训练与消融）。

**演示模式**：在“标注”页面点击「生成演示图像」，即可在没有真实数据、没有 API Key 的情况下完整体验标注、清洗、数据集划分与训练看板全流程。

### 运行测试

```bash
pip install -r requirements-dev.txt
pytest -v
```

---

## 目录结构

```text
LabelAgent/
├─ README.md                        # 项目说明（本文件）
├─ README_labelagent.md             # 桌面版详细用户手册（Release v1.0.0）
├─ LICENSE                          # MIT 许可证
├─ pyproject.toml                   # 包元数据与工具配置
├─ requirements.txt                 # 运行依赖
├─ requirements-dev.txt             # 开发/测试依赖
├─ run.py                           # 启动入口
├─ labelagent/                      # 核心源码包
│  ├─ cli.py                        #   命令行入口（labelagent 命令）
│  ├─ config.py                     #   全局配置与默认参数
│  ├─ demo.py                       #   演示图像生成（合成数据）
│  ├─ core/                         # 数据模型与项目状态
│  │  ├─ models.py                  #   Annotation / ImageItem 等数据模型
│  │  └─ store.py                   #   ProjectStore：内存项目状态（线程安全）
│  ├─ annotation/                   # 🏷️ 标注模块
│  │  ├─ importer.py                #   图像导入（文件夹/多文件）
│  │  ├─ agent.py                   #   Agent 自动标注调度与进度
│  │  ├─ providers.py               #   标注 Provider：本地 DINO+SAM2 / 在线大模型 / Demo
│  │  ├─ editor.py                  #   标注修订（重命名/删除/更新）
│  │  └─ export.py                  #   五格式导出 + ZIP 打包
│  ├─ cleaning/                     # 🧹 清洗模块
│  │  ├─ quality.py                 #   质量诊断：模糊/空图/健康得分
│  │  ├─ duplicate.py               #   重复图检测：dHash + LSH / Pairwise + 缓存
│  │  ├─ bounds.py                  #   越界标注检查与纠偏
│  │  └─ cleaner.py                 #   自动清洗编排
│  ├─ dataset/                      # 📂 数据集模块
│  │  ├─ stats.py                   #   类别统计
│  │  └─ split.py                   #   Train/Val 划分 + dataset.yaml 生成
│  ├─ environment/                  # 🐍 环境模块
│  │  └─ python_env.py              #   Python 解释器检测/导入/依赖包查看
│  ├─ training/                     # 🚀 训练与消融模块
│  │  ├─ config.py                  #   模型/Loss/优化器/调度/策略/两阶段配置
│  │  ├─ scriptgen.py               #   生成外部 Python 训练脚本
│  │  ├─ runner.py                  #   训练执行器（外部子进程 / 演示模式）
│  │  ├─ monitor.py                 #   训练看板状态（进度/Loss/日志）
│  │  └─ ablation.py                #   消融实验管理与 CSV 导入对比
│  ├─ api/                          # FastAPI 后端
│  │  ├─ app.py                     #   应用工厂 + 静态资源托管
│  │  ├─ annotation_routes.py       #   标注相关接口
│  │  ├─ cleaning_routes.py         #   清洗相关接口
│  │  ├─ dataset_routes.py          #   数据集相关接口
│  │  ├─ environment_routes.py      #   环境相关接口
│  │  ├─ training_routes.py         #   训练/消融相关接口
│  │  └─ demo_routes.py             #   演示图像生成接口
│  └─ web/                          # Web 前端（桌面版 WebView 同源页面）
│     ├─ index.html
│     ├─ css/style.css
│     └─ js/app.js
├─ docs/                            # 说明文档
│  ├─ architecture.md               # 源码架构说明
│  ├─ api.md                        # HTTP 接口说明
│  └─ development.md                # 开发指南
├─ tests/                           # 单元测试（pytest）
│  ├─ conftest.py
│  ├─ test_annotation_export.py
│  ├─ test_cleaning.py
│  ├─ test_dataset.py
│  ├─ test_training_config.py
│  └─ test_ablation.py
└─ workspace/                       # 运行期工作目录（自动创建，git 忽略）
   ├─ demo_images/                  #   演示图像
   ├─ exports/                      #   导出文件
   ├─ cache/                        #   哈希缓存
   ├─ runs/                         #   训练运行记录
   └─ labelagent.log                #   运行日志
```

---

## 源码架构

LabelAgent 采用 **本地服务 + Web 界面** 的分层架构（桌面版通过 WebView 加载同一套前端页面）：

```text
┌─────────────────────────────────────────────────┐
│  Web 前端 (labelagent/web)                      │
│  标注 | 清洗 | 数据集 | 环境 | 训练与消融         │
└──────────────────────┬──────────────────────────┘
                       │ HTTP / JSON
┌──────────────────────▼──────────────────────────┐
│  API 层 (labelagent/api) — FastAPI               │
│  校验请求 → 调用业务层 → 返回 JSON               │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  业务层 (labelagent/*)                           │
│  annotation │ cleaning │ dataset │ environment │ │
│  training │ ablation                             │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  数据层                                             │
│  ProjectStore（内存状态）│ 文件系统（workspace/）  │
│  外部 Python 训练环境（子进程）│ 在线模型 API      │
└─────────────────────────────────────────────────┘
```

各层职责：

- **数据模型层**（`core/`）：`Annotation`、`ImageItem`、训练/消融配置等 Pydantic 模型，以及线程安全的 `ProjectStore` 作为整个应用的内存状态中心；
- **业务层**（`annotation/`、`cleaning/`、`dataset/`、`environment/`、`training/`）：纯 Python 实现，不依赖 Web 框架，可独立测试与复用；
- **API 层**（`api/`）：FastAPI 路由，负责请求校验与状态编排，并提供静态资源托管；
- **外部集成**：训练引擎通过子进程调用外部 Python 环境执行生成的训练脚本；在线标注模型通过 HTTP 调用各家 API。

详细说明见 [docs/architecture.md](./docs/architecture.md)。

---

## 接口说明

后端提供 RESTful 接口，统一前缀 `/api`：

| 分组 | 说明 |
|---|---|
| `/api/annotation/*` | 图像导入、Agent 标注、标注修订、格式导出 |
| `/api/cleaning/*` | 质量诊断、自动清洗、哈希缓存管理 |
| `/api/dataset/*` | 类别统计、数据集划分、dataset.yaml |
| `/api/environment/*` | Python 环境检测、导入、依赖包查询 |
| `/api/training/*` | 训练配置、启动/停止、看板状态、产物下载 |
| `/api/ablation/*` | 消融实验组管理、批量运行、CSV 导入、结果导出 |
| `/api/demo/*` | 演示图像生成 |

完整接口清单与请求/响应示例见 [docs/api.md](./docs/api.md)。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [README_labelagent.md](./README_labelagent.md) | 桌面版（Release v1.0.0）完整用户手册：功能说明、使用流程、常见问题 |
| [docs/architecture.md](./docs/architecture.md) | 源码架构说明：分层设计、核心流程、模块职责 |
| [docs/api.md](./docs/api.md) | HTTP 接口说明：接口清单、参数与示例 |
| [docs/development.md](./docs/development.md) | 开发指南：环境搭建、代码组织、测试、打包思路 |

---

## 已知事项与后续规划

- **标注微调**：当前版本支持标签重命名与删除；完整的手工绘制/拖拽顶点工具在后续版本提供。
- **模糊阈值控件**：用户手册中记录过 `cleanBlurThresh` 参数缺失问题；源码版已在清洗页面提供「模糊度阈值」输入控件，并绑定到诊断逻辑。
- **训练环境**：源码版默认以「演示模式」运行训练看板（无需 GPU）；接入真实训练需在「环境」模块导入装有 PyTorch / Ultralytics 等依赖的 Python 解释器，并选择对应模型架构。
- **模型权重**：本地 Grounding DINO + SAM 2 的权重与依赖需用户自行准备，本项目不内置大型模型权重。
- **路线图**：手工标注编辑器、数据增强、更多训练引擎适配（Detectron2 / MMDetection）、WebSocket 实时看板。

---

## 许可证

[MIT](./LICENSE)

---

*说明：本仓库 `README.md` 与 `README_labelagent.md` 分别面向源码开发者和桌面版用户；源码版功能与桌面版保持对齐，并补充了演示模式以便无环境快速体验。*
