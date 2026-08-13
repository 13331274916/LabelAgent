# LabelAgent HTTP 接口说明

后端为 FastAPI 应用，统一前缀 `/api`，响应结构：

```json
{ "ok": true, "data": { ... } }       // 成功
{ "ok": false, "error": "说明" }      // 失败（HTTP 400/403/404 等）
```

Web 静态页面挂在根路径 `/`（桌面版 WebView 加载同一套页面）。启动后可在
`http://127.0.0.1:8765/docs` 查看自动生成的 OpenAPI 文档（FastAPI 自带）。

## 1. 通用

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 服务健康检查 |
| GET | `/api/overview` | 项目概览（图像数 / 标注数 / 类别数） |

## 2. 标注 `/api/annotation`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/providers` | 标注 Provider 列表（含可用性） |
| POST | `/import-folder` | 导入文件夹 `{folder, recursive?}` |
| POST | `/import-paths` | 按路径列表导入 `{paths: [...]}` |
| POST | `/upload` | 浏览器上传图像（multipart `files`） |
| GET | `/images` | 图像列表（摘要） |
| GET | `/image/{id}` | 图像详情（含标注） |
| GET | `/image/{id}/file` | 图像文件 |
| POST | `/agent/start` | 启动 Agent 标注 `{provider_id, prompt, label, max_objects, api_key?, image_ids?}` |
| GET | `/agent/status` | Agent 标注任务状态 |
| POST | `/image/{id}/annotations` | 新增标注 |
| PATCH | `/image/{id}/annotations/{ann_id}` | 更新标注 `{label?, points?, confidence?}` |
| DELETE | `/image/{id}/annotations/{ann_id}` | 删除标注 |
| POST | `/image/{id}/export` | 单文件导出 `{format: labelme\|voc\|yolo\|coco\|csv}` |
| POST | `/export-zip` | ZIP 批量导出 `{formats: [...], filename?}` |
| GET | `/download?path=...` | 下载工作目录内的文件 |

## 3. 清洗 `/api/cleaning`

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/diagnose` | 质量诊断 `{blur_threshold?, color_std_threshold?}` → 健康得分与异常统计 |
| POST | `/auto-clean` | 自动清洗 `{remove_duplicates?, fix_oob?, remove_blurry?, remove_empty?, duplicate_method?}` |
| GET | `/cache` | 哈希缓存信息 |
| DELETE | `/cache` | 清空哈希缓存 |

## 4. 数据集 `/api/dataset`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/stats` | 类别统计 |
| POST | `/split` | 数据集划分 `{train_ratio: 0.50~0.95, dataset_name}` → Train/Val 数量与 dataset.yaml 路径 |
| GET | `/yaml?path=...` | 下载 dataset.yaml |
| GET | `/datasets` | 已生成的数据集列表 |

## 5. 环境 `/api/environment`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/detect` | 检测本机 Python 解释器 |
| POST | `/import` | 手动导入 `{path: python.exe}` |
| POST | `/packages` | 读取指定解释器的依赖包列表 |

## 6. 训练与消融 `/api/training`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/options` | 模型 / Loss / 优化器 / 调度器 / 策略选项 |
| POST | `/start` | 启动训练 `{config: TrainingConfig, mode: demo\|external, dataset_yaml?}` |
| GET | `/status` | 训练看板状态（进度 / Loss / 日志 / 产物） |
| POST | `/stop` | 停止训练 |
| GET | `/artifacts` | 可下载产物列表（results.csv / best*.pt） |
| GET | `/ablation/groups` | 消融实验组列表 |
| POST | `/ablation/groups` | 新增实验组 `{name, model_arch, warmup?, ema?, mosaic?, ...}` |
| DELETE | `/ablation/groups/{id}` | 删除实验组 |
| POST | `/ablation/run` | 批量运行消融实验 `{mode?, dataset_yaml?}` |
| GET | `/ablation/status` | 消融运行状态 + 结果 |
| POST | `/ablation/import-csv` | 批量导入已有 results.csv（multipart `files` + `base_name`） |
| GET | `/ablation/summary.csv` | 下载 ablation_summary.csv |
| GET | `/ablation/plot.png` | 下载消融对比图 |
| GET | `/download?path=...` | 下载工作目录内的文件 |

## 7. 演示 `/api/demo`

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/generate` | 生成合成演示图像 `{count?, width?, height?}` 并导入项目 |

## 8. 调用示例

```bash
# 生成演示图像并自动标注（demo provider，无需 API Key）
curl -X POST http://127.0.0.1:8765/api/demo/generate \
     -H 'Content-Type: application/json' -d '{"count": 12}'
curl -X POST http://127.0.0.1:8765/api/annotation/agent/start \
     -H 'Content-Type: application/json' \
     -d '{"provider_id": "demo", "prompt": "检测划痕缺陷", "label": "scratch_defect", "max_objects": 5}'

# 质量诊断与自动清洗
curl -X POST http://127.0.0.1:8765/api/cleaning/diagnose \
     -H 'Content-Type: application/json' -d '{"blur_threshold": 100}'
curl -X POST http://127.0.0.1:8765/api/cleaning/auto-clean \
     -H 'Content-Type: application/json' -d '{"remove_duplicates": true, "fix_oob": true}'

# 数据集划分
curl -X POST http://127.0.0.1:8765/api/dataset/split \
     -H 'Content-Type: application/json' -d '{"train_ratio": 0.8, "dataset_name": "steel_defect_v1"}'

# 训练（演示模式）
curl -X POST http://127.0.0.1:8765/api/training/start \
     -H 'Content-Type: application/json' \
     -d '{"mode": "demo", "config": {"model_arch": "RT-DETR", "loss": "Focal Loss + EIoU", "epochs": 12, "stage1_epochs": 8, "stage2_epochs": 4}}'
```
