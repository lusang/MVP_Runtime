# 预标注就绪检查 — 20 条测试数据 + 人工标注对比

> 检查时间：2026-05-26  
> 检查人：MVP Runtime 自检  
> 测试规模：20 张图片  
> 标记类型：人工标注已存在，运行结束后对比

---

## 1. 环境配置

| 检查项 | 状态 | 说明 |
|--------|------|------|
| `MVP_HOST` | 0.0.0.0 | 允许外部访问 |
| `MVP_PORT` | 8001 | 默认端口 |
| `MVP_FORCE_YOLO_MOCK` | 0 | YOLO 真实推理（CPU） |
| `MVP_FORCE_GEMINI_MOCK` | **0** ✅ | 真实 Gemini API |
| `MVP_DISABLE_PLANNER` | 1 | 使用确定性 Planner |
| `GEMINI_API_KEY` | 已配置 ✅ 实测有效 | 2026-05-26 验证通过 |
| `GEMINI_MODEL` | gemini-2.0-flash | 默认模型 |
| YOLO 权重文件 | 25 MB `yolov8s-worldv2.pt` | 存在 |
| YOLO 设备 | CPU | 20 张图片可接受 |

### ⚠️ 已解决：Gemini 现为真实模式

2026-05-26 验证：
- `MVP_FORCE_GEMINI_MOCK=0` ✅
- GEMINI_API_KEY 实测可通 ✅（`gemini-2.0-flash` 返回正常）
- 全链路管道执行无错误 ✅

---

## 2. API 服务就绪

| 端点 | 方法 | 用途 | 状态 |
|------|------|------|------|
| `/health` | GET | 健康检查 | ✅ 已实现 |
| `/run_annotation` | POST | 单图同步标注 | ✅ 已实现 |
| `/run_annotation_async` | POST | 批量异步标注 | ✅ 已实现 |

服务器启动命令：

```bash
cd /path/to/MVP_Runtime-main
python main.py
# → uvicorn on 0.0.0.0:8001
```

---

## 3. 异步批处理链路

### 数据流

```
DataHub → POST /run_annotation_async → 202 Accepted
  → async_worker.process_batch()
    → 解析 template（从请求体嵌入的 JSON）
    → Planner.compile(parsed)     # 编译一次
    → 遍历 tasks:
      → 遍历 frames:
        → resolve_url(frame.url)  # file:// 直接读 / http:// 下载
        → engine.run_with_plan()  # 推理
        → 收集结果
    → 每完成一个 task → POST callback_url（带重试）
    → 清理临时目录
```

### 回调重试策略

| 维度 | 值 |
|------|-----|
| 重试次数 | 最多 3 次 |
| 延迟序列 | [0s, 5s, 30s] |
| 超时 | 30s |
| 最终兜底 | 写入 `storage/dead_letter/{run_id}/{task_id}.json` |

### 文件 URL 支持

| 协议 | 说明 |
|------|------|
| `file://` | 直接提取本地路径，零拷贝 |
| `http(s)://` | 下载到临时目录，用完清理 |

---

## 4. 历史运行记录

### 死信队列（callback 失败）

`storage/dead_letter/` 下有 **11 个 run_id** 目录，含大量 `.json` 文件。

**关键发现**：所有死信文件的 `frames[].status = "completed"`（管道执行成功），失败原因均为 **callback URL 不可达**（`http://datahub:8000/` 在当前网络不存在）。

**结论**：管道执行本身无问题，回调地址需确认可用。

### 运行时数据库

| 数据库 | 表 | 状态 |
|--------|-----|------|
| `storage/runtime_events.db` | runtime_run, runtime_step, runtime_step_event | 存在，有历史记录 |
| `storage/performance.db` | performance_log | 存在，有历史记录 |

---

## 5. 模板匹配检查

DataHub 发送的 `template` 是嵌入在请求体中的独立 JSON，**不是** `resource/Template.json`。

**历史运行中使用的模板字段来自死信记录：**

| 字段 | 值 |
|------|-----|
| `object_name` | `"objects"` |
| 语义属性 | `object_type`（multi_select）, `is_package`（boolean） |
| 质量属性 | `background_clutter`（boolean） |
| 负属性 | `pure_negative`, `ambiguous`, `open_set_negative` |

**资源模板 `resource/Template.json`：**

| 字段 | 值 |
|------|-----|
| `object_name` | `"Package"` |
| 语义属性 | `package_form`, `brand_logo`, `size_category` |
| 质量属性 | `occlusion`, `blur`, `lighting` |
| 负属性 | `Pure Negative`, `Hard Negative`, `Ambiguous`, `Open-set Negative` |

**⚠️ 两个模板不一致**。如果 20 张测试数据的人工标注使用 `resource/Template.json` 的 schema，则 DataHub 发送的 template 必须与之匹配。否则 AI 输出与人工标注无法直接对比。

---

## 6. 需要确认的事项（阻塞项）

### 必须确认

| # | 事项 | 原因 |
|---|------|------|
| 1 | **Gemini 是否要用真实 API？** | `MVP_FORCE_GEMINI_MOCK=1` → mock 结果无法与人工标注对比 |
| 2 | **20 张图片的模板 schema 是什么？** | 需要匹配 20 张测试数据的 schema（`object_name`、属性列表） |
| 3 | **DataHub 回调地址是什么？** | 地址不可达则结果写入死信队列 |
| 4 | **图片路径协议？** | DataHub 通过 `file://` 还是 `http://` 传递图片 |
| 5 | **DataHub 发送的请求体中 `template` 字段？** | 模板由 DataHub 嵌入请求体 |

### 建议操作

```bash
# 1. 关闭 Gemini Mock（如需真实推理）
# 编辑 config/.env
MVP_FORCE_GEMINI_MOCK=0

# 2. 启动服务器
python main.py

# 3. 验证健康检查
curl http://localhost:8001/health
# → {"status": "ok", "service": "mvp-runtime"}

# 4. 验证 API Key（可选）
python -c "from models.gemini_client import _get_genai_client; c=_get_genai_client('AIza...', 30); print('client created')"
```

---

## 7. 测试流程预期

```
DataHub
  │
  ├─ ① 对每个 task（每张图一个 task），按 API_INTEGRATION.md 格式
  │    POST → MVP /run_annotation_async
  │
  ├─ ② MVP 处理：
  │    每帧运行完整 6 步 pipeline
  │    （detect → verify → quality → semantic → negative → merge）
  │
  ├─ ③ 每完成一个 task → POST 回调到 DataHub
  │    回调体包含 annotation_result（清洁格式）+ runtime_trace（调试格式）
  │
  └─ ④ DataHub 收到回调后：
       存储 AI 标注结果
       与已有的人工标注进行逐帧对比
       （对比工具和指标由 DataHub 侧定义，MVP 不提供）
```

**AI 输出 vs 人工标注对比维度建议：**

| 维度 | 对比内容 |
|------|---------|
| Detection | bbox IoU / mAP / 漏检率 / 误检率 |
| Classification | is_positive 一致性（精确率/召回率） |
| Attributes | 每个语义属性的值一致性 |
| Confidence | AI 置信度与 human-agreement 的相关性 |
| Latency | 每张图处理耗时 |
