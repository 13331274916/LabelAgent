# LabelAgent 源码架构说明

本文档面向源码阅读者与二次开发者，介绍 LabelAgent 的分层设计、核心流程与各模块职责。

## 1. 总体架构

LabelAgent 采用 **本地服务 + Web 界面** 的分层架构：

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
│  annotation │ cleaning │ dataset │ environment  │
│  training │ ablation                             │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  数据层                                             │
│  ProjectStore（内存状态）│ 文件系统（workspace/）  │
│  外部 Python 训练环境（子进程）│ 在线模型 API      │
└─────────────────────────────────────────────────┘
```

设计目标：

1. **业务层与 Web 框架解耦**：所有核心逻辑（导出格式、清洗算法、数据集划分等）是纯 Python 函数，不依赖 FastAPI，可独立单元测试、可在其他界面（如桌面 WebView、CLI、脚本）中复用；
2. **单一状态中心**：`ProjectStore` 持有全部图像与标注，各模块通过它读写数据，保证并发安全；
3. **外部能力可插拔**：标注 Provider 与训练执行器都是接口化的，方便接入新的模型厂商与训练框架。

## 2. 模块职责

### 2.1 数据模型层 `labelagent/core/`

| 文件 | 职责 |
|---|---|
| `models.py` | Pydantic 数据模型：`Point` / `Annotation`（多边形标注）、`ImageItem`（图像 + 标注 + 清洗标记）、`TrainingConfig`（训练配置）、`AblationGroup` / `AblationMetrics`（消融）等 |
| `store.py` | `ProjectStore`：线程安全的内存项目状态，提供图像的增删查改、标注修订、类别列表与快照 |

`Annotation.bbox` / `clamp()` 等属性方法集中定义了标注几何语义，导出、越界检查、前端渲染均复用它。

### 2.2 标注模块 `labelagent/annotation/`

| 文件 | 职责 |
|---|---|
| `importer.py` | 图像导入：文件夹扫描、多文件导入、尺寸读取（PIL） |
| `providers.py` | 标注 Provider 体系：`BaseProvider` 抽象基类 + 本地 Grounding DINO + SAM 2、DeepSeek-VL、Qwen2-VL、GPT-4o、Demo 五种实现；`PROVIDER_REGISTRY` 工厂注册表 |
| `agent.py` | `AgentAnnotator`：后台线程批量执行标注，维护任务状态（进度、日志） |
| `editor.py` | 标注修订：标签重命名、删除、顶点更新、手工新增 |
| `export.py` | 五格式导出：LabelMe JSON / VOC XML / YOLO TXT / COCO JSON / CSV，单文件导出与 ZIP 批量打包 |

**Provider 接口**：

```python
class BaseProvider:
    def annotate(self, image_path, prompt, label, max_objects,
                 api_key=None, **kwargs) -> tuple[list[Annotation], str]: ...
    def availability(self) -> tuple[bool, str]: ...
```

在线模型（DeepSeek-VL / Qwen2-VL / GPT-4o）共享 `_OnlineChatProvider`：把图像 base64 编码后发给 OpenAI 兼容的 `chat/completions` 接口，再通过 `parse_annotation_json()` 把模型输出的 JSON 解析为标注（兼容 ```` ```json ``` ```` 包裹）。

### 2.3 清洗模块 `labelagent/cleaning/`

| 文件 | 职责 |
|---|---|
| `quality.py` | 质量诊断：Laplacian 方差模糊检测、纯色空图检测、健康得分（0~100，按异常项扣分） |
| `duplicate.py` | 重复图检测：dHash 感知哈希 + LSH 分桶（O(N)）/ Pairwise（O(N²)）+ `HashCache` JSON 持久化缓存 |
| `bounds.py` | 越界标注检查（容差规则）与自动纠偏（顶点夹紧回图像范围） |
| `cleaner.py` | `auto_clean()` 自动清洗编排：剔除重复/模糊/空图、纠偏越界框 |

### 2.4 数据集模块 `labelagent/dataset/`

| 文件 | 职责 |
|---|---|
| `stats.py` | 类别统计（图像数 / 标注数 / 每类数量） |
| `split.py` | Train/Val 划分：比例校验（0.50~0.95，步长 0.05）、随机划分（固定种子可复现）、复制图像 + 导出 YOLO 标签 + 生成 `dataset.yaml` |

### 2.5 环境模块 `labelagent/environment/`

`python_env.py`：发现本机 Python 解释器（当前进程 / PATH / conda 常见路径）、手动导入外部解释器、通过 `importlib.metadata` 读取依赖包列表并标记训练相关包。训练相关的大型框架（PyTorch 等）不打包进 LabelAgent，由该模块对接外部环境。

### 2.6 训练模块 `labelagent/training/`

| 文件 | 职责 |
|---|---|
| `config.py` | 训练选项清单（模型架构 / Loss / 优化器 / 调度器 / 策略），来自 `labelagent/config.py` 常量 |
| `monitor.py` | `TrainingMonitor`：训练看板状态（进度、当前 epoch、Loss 历史、控制台日志），线程安全 |
| `runner.py` | `TrainingRunner`：训练执行器。`demo` 模式在进程内模拟两阶段训练（确定性、可复现）；`external` 模式调用外部 Python 子进程执行生成的训练脚本，解析 JSON 行进度 |
| `scriptgen.py` | 把 `TrainingConfig` 渲染为自包含的外部训练脚本（含框架可用性检查、进度上报协议、results.csv 与权重产物输出协议） |
| `ablation.py` | `AblationManager`：实验组管理、批量运行、已有 results.csv 导入、mAP@0.5 最佳标记与相对提升计算、summary CSV 与 matplotlib 对比图导出 |

### 2.7 API 层 `labelagent/api/`

FastAPI 应用工厂 `app.py` 创建 `AppState`（共享 store / cache / runner / ablation），挂载各路由与 Web 静态资源。路由按模块拆分，全部返回 `{ok: true, data: ...}` 或 `{ok: false, error: ...}` 统一结构。

## 3. 核心流程

### 3.1 Agent 自动标注流程

```text
前端提交 provider_id + 提示词 + 标签规则
        ↓
AgentAnnotator.start() 启动后台线程
        ↓
对每张图像调用 Provider.annotate()
   ├─ demo：基于图像内容哈希生成确定性伪标注
   ├─ online：base64 图像 → chat/completions → 解析 JSON → Annotation
   └─ local：依赖本地 torch/groundingdino/sam2 环境（需自行准备）
        ↓
store.set_annotations(image_id, annotations) 写回内存状态
        ↓
前端轮询 /api/annotation/agent/status 刷新进度与图库
```

### 3.2 数据清洗流程

```text
diagnose()：模糊检测 + 空图检测 + 重复图检测(LSH/Pairwise) + 越界检查
   ↓ 更新 ImageItem 标记（is_blurry / is_empty / duplicate_of / has_oob）
   ↓ 汇总 DiagnosticResult（健康得分、各类异常数量）
auto_clean()：按勾选项执行 剔除重复 → 纠偏越界 → 剔除模糊/空图
   ↓ 返回 CleanResult（保留/移除/修复统计）
```

### 3.3 训练与消融流程

```text
TrainingConfig 校验 → runner.start()
   ├─ demo：内置确定性模拟训练（每 epoch 上报 Loss）
   └─ external：生成 train_script.py + config.json → 子进程执行
        （脚本按 {json 行} 上报进度，由 monitor 记录）
        ↓
训练结束：results.csv + best_stage*.pt + best.pt
        ↓
AblationManager：多组配置批量运行 → 汇总 mAP@0.5 / Val Loss / Precision
   → 最佳标记 + 相对提升 → ablation_summary.csv + 对比图
```

## 4. 关键设计说明

- **演示模式（demo）**：为了让没有 GPU / API Key 的用户完整体验全流程，标注提供 `demo` Provider，训练提供 `demo` 模式；所有演示产物都明确标注 `"demo": true`，不会与真实权重混淆。
- **诚实的外部训练接口**：`scriptgen.py` 生成的脚本会先检查目标环境是否具备框架（YOLO 需要 ultralytics），缺少时输出明确错误并退出，绝不“假装训练成功”。
- **路径安全**：所有下载接口都校验解析后的路径必须位于 `workspace/` 内，防止目录穿越。
- **可测试性**：业务层全部为纯函数/类，不依赖网络与 GUI，22 个单元测试覆盖导出格式、清洗算法、数据集划分、训练配置与消融汇总。
