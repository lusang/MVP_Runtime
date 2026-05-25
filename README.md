# MVP Runtime — Vision Agent Pre-annotation Runtime

## 架构概览

```
HTTP 请求 (template + tasks)
        │
        ▼
┌─────────────────────────────────────────────────┐
│                  FastAPI App                      │
│  api/routes.py                                    │
│    GET  /health                                   │
│    POST /run_annotation                           │
│    POST /run_annotation_async  →  202 Accepted    │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│              RuntimeEngine                        │
│  runtime/engine.py                                │
│    run() → parse template → compile plan → exec   │
│    run_with_plan() → execute pre-compiled plan    │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│              StepExecutor                         │
│  runtime/step_executor.py                         │
│   按 plan.steps[].order 排序，逐步骤执行           │
│   支持 early_exit / skip_conditions               │
└──┬───┬───┬───┬───┬───┬───┬────────────────────┘
   │   │   │   │   │   │   │
   ▼   ▼   ▼   ▼   ▼   ▼   ▼
  步骤1~6 (见下方)
```

---

## Pipeline 步骤详解

### Step 0: detect — YOLO 检测

**文件：**
- `models/yolo_detector.py` — 适配器层，调用 backend
- `models/yolo_backend.py` — 实际推理（Ultralytics YOLO 或 mock）
- `models/object_target.py` — 从 template 提取目标描述
- `models/world_vocab.py` — 构建 YOLO-World set_classes 词汇表

**流程：**
```
template.objects[0].name（如 "包裹"）
  → ObjectTarget.from_parsed(parsed)
  → world_classes_from_target(target)
    → 1. target.name 原样加入（可能是中文）
    → 2. 如果 name != "package"，额外加 "package"
    → 3. 解析 include 字段（逗号/分隔符分割）
    → 4. 追加硬编码英文默认词 ["package", "cardboard box", ...]
  → model.set_classes(classes)   ← 中英文混合传给 YOLO-World
  → model.predict(image_path)
  → crop 每个 bbox 到 storage/preselection/{run_id}/
```

**关键点：** YOLO-World 是 open-vocab 模型，`set_classes()` 接受任意文本。中文词汇可直接传入，但建议加英文兜底。当前实现在 `world_vocab.py` 中已经做了中英混合。

---

### Step 1: verify — Gemini 验证

**文件：**
- `handlers/verification_handler.py` — 轻量 façade
- `models/gemini_verifier.py` — 路由到 real Gemini 或 mock
- `models/gemini_client.py` — 实际 API 调用 + 提示词

**提示词模板** (`gemini_client.py:46`):
```
_VERIFY_OBJECT_PROMPT = """\
You are a precision vision annotation agent.
Task: determine if the provided image crop contains a "{object_name}".

Target definition: {description}
Include (positive indicators): {include}
Exclude (do NOT count these): {exclude}

Respond ONLY with a valid JSON object:
{"ok": true or false, "score": 0.0 to 1.0, "rationale": "..."}"""
```

**填入内容：** template 中的原始中文值（`object_name="包裹"`, `include="包裹, 快递盒..."`）

**输入：** 每个候选框的 crop 图
**输出：** `{ok, score, rationale}`

---

### Step 2: quality — OpenCV 质量分析

**文件：** `models/opencv_analyzer.py`

**非 AI，纯图像处理算法：**

| 属性名 | 算法 | 判断依据 |
|--------|------|---------|
| blur | Laplacian 方差 | >150 清晰, >50 轻微, <50 重度模糊 |
| lighting | 直方图均值 | >180 过曝, <60 偏暗, 中间正常 |
| occlusion | Canny 边缘密度 | >0.08 无遮挡, >0.03 部分, <0.03 严重 |

**输入：** crop 图的灰度区域
**输出：** `{value, confidence, metrics}`

**注意：** 当 cv2 不可用时，自动 fallback 到 mock 返回值。

---

### Step 3: attribute — Gemini 语义属性

**文件：**
- `handlers/plugins/gemini_attribute.py` — plugin 实现
- `models/gemini_client.py:_VERIFY_ATTRIBUTE_PROMPT` (L58)

**提示词模板：**
```
_VERIFY_ATTRIBUTE_PROMPT = """\
You are a precision vision annotation agent.
For this image crop containing a "{object_name}", determine: {attribute_name}

Attribute description: {description}
Type: {attribute_type}
ALLOWED OPTIONS: {options}

CRITICAL ENUM CONSTRAINT:
- You MUST choose ONLY from the ALLOWED OPTIONS list above.
- If NONE of the allowed options convincingly match → return null.

Respond ONLY with:
{"value": <exact option or null>, "confidence": 0.0 to 1.0}"""
```

**每个属性独立调用一次 Gemini**（如"颜色"、"有快递单"各一次）。

**后处理：** `_validate_attribute_value()` 校验返回值是否在 options 内，自动修正大小写和过滤非法值。

---

### Step 4: negative — Gemini 负样本检查

**文件：**
- `handlers/plugins/gemini_negative.py` — plugin 实现
- `models/gemini_client.py:_VERIFY_NEGATIVE_ATTRIBUTE_PROMPT` (L77)

与 semantic 属性类似，但：
- **使用全图**而非 crop（能看到场景上下文）
- 检查"硬负样本 / 纯负样本 / 开放集负样本"等

---

### Step 5: merge — Gemini 合并审核

**文件：**
- `models/gemini_merger.py` — 调用 Gemini 或 mock
- `models/gemini_client.py:_MERGE_PROMPT` (L115)

**特殊：这是唯一不看图片的步骤。** 只输入完整执行日志文本。

**提示词模板关键内容：**
```
YOUR ROLE — strict decision logic only:
1. Scan the log for conflicts between steps
2. Resolve conflicts by applying decision rules
3. Normalize confidence scores into merge_confidence
4. Produce the final annotation panel
```

**决策规则：**
- verification 拒绝 → 负样本
- 任何 hard negative flag 触发 → 负样本
- Pure Negative 场景检查通过 → 所有候选框负样本
- detector + verification 都接受 → 正样本

**当 `MVP_FORCE_GEMINI_MOCK=1`（当前 .env 设置）时：** 走 mock 合并，机械平均置信度，不调用真实 Gemini API。

---

## 提示词位置汇总

| 步骤 | 模型 | 提示词位置 | 语言 | 是否可配置 |
|------|------|-----------|------|-----------|
| detect | YOLO-World | `world_vocab.py:world_classes_from_target()` | 中+英混合 | 改代码 / YOLO_WORLD_EXTRA_CLASSES |
| verify | Gemini | `gemini_client.py:_VERIFY_OBJECT_PROMPT` (L46) | 中文（模板原样） | 改代码 |
| quality | OpenCV | `opencv_analyzer.py`（算法，无提示词） | — | 不可配置 |
| semantic attribute | Gemini | `gemini_client.py:_VERIFY_ATTRIBUTE_PROMPT` (L58) | 中文 | 改代码 |
| negative attribute | Gemini | `gemini_client.py:_VERIFY_NEGATIVE_ATTRIBUTE_PROMPT` (L77) | 中文 | 改代码 |
| scene negative | Gemini | `gemini_client.py:_VERIFY_SCENE_NEGATIVE_PROMPT` (L101) | 中文 | 改代码 |
| merge | Gemini | `gemini_client.py:_MERGE_PROMPT` (L115) | 中文+英文 | 改代码 |

---

## Planner 系统

**文件：** `runtime/planner.py`

### 当前状态：静态计划（`MVP_DISABLE_PLANNER=1`）

走 `_StaticPlanFactory.build(parsed)`，产生固定 6 步计划（detect → verify → quality → attribute → negative → merge）。

### 启用动态 Planner（`MVP_DISABLE_PLANNER=0`）

会调用 Gemini Planner，让它根据：
- 模板内容（有哪些属性、负样本等）
- 模型目录（model_registry）
- 历史性能数据（performance_tracker）

动态决定：
- 步骤顺序
- 每个步骤用哪个模型
- early_exit_rules（提前退出条件）
- skip_conditions（跳过条件）

---

## 异步批处理（POST /run_annotation_async）

**文件：**
- `api/routes.py` — 端点定义，返回 202 + run_id
- `runtime/async_worker.py` — 后台编排
- `runtime/callback.py` — 回调重试（3 次：0s → 5s → 30s）+ 死信队列
- `storage/url_resolver.py` — 支持 file:// 和 http(s)://

**流程：**
```
请求 → 202 Accepted (immediate)
  ↓ asyncio.create_task
后台处理每个 task：
  ├─ 解析 template → ParsedTaskSpec
  ├─ Planner.compile(parsed) → 共享计划
  ├─ 遍历 frames：
  │   ├─ resolve URL（file:// 直接读 / http:// 下载）
  │   └─ engine.run_with_plan(image, plan, parsed)
  └─ 每完成一个 task → POST callback_url（重试 3 次）
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MVP_HOST` | `0.0.0.0` | 服务绑定地址 |
| `MVP_PORT` | `8001` | 服务端口 |
| `MVP_FORCE_YOLO_MOCK` | `0` | 1=使用 mock 检测框 |
| `MVP_FORCE_GEMINI_MOCK` | `1` | 1=所有 Gemini 步骤走 mock |
| `MVP_DISABLE_PLANNER` | `1` | 1=使用静态计划 |
| `YOLO_MODEL_PATH` | `weights/yolov8s-worldv2.pt` | YOLO 权重路径 |
| `YOLO_CONF_THRESHOLD` | `0.25` | 检测置信度阈值 |
| `YOLO_DEVICE` | `cpu` | 推理设备 |
| `YOLO_WORLD_EXTRA_CLASSES` | — | 额外 YOLO 提示词（逗号分隔） |
| `YOLO_WORLD_MAX_CLASSES` | `12` | 最大提示词数量 |
| `GEMINI_API_KEY` | — | Google AI API 密钥 |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini 模型 ID |
| `GEMINI_TIMEOUT_SEC` | `120` | API 超时秒数 |

---

## Mock 机制

当前 `.env` 中 `MVP_FORCE_GEMINI_MOCK=1`，所有 Gemini 步骤返回模拟数据，不调用真实 API。

**各模型的 mock 行为：**

| 步骤 | Mock 行为 |
|------|----------|
| YOLO | `_mock_candidates()` — 按图像比例放两个 bbox |
| Gemini verify | 返回 `ok=True, score=0.88` |
| Gemini attribute | 返回 `value=options[0], confidence=0.85` |
| Gemini merge | `_mock_merge()` — 机械平均置信度 + 固定 trace |
| OpenCV quality | 返回 `value=options[0], confidence=0.8`（无 cv2 时） |

---

## 扩展 / 修改指南

### 修改 YOLO 提示词
- 环境变量 `YOLO_WORLD_EXTRA_CLASSES=box,carton,crate`
- 或修改 `world_vocab.py:_DEFAULT_EN` 常量
- 或修改 `world_classes_from_target()` 逻辑

### 修改 Gemini 提示词
所有提示词在 `models/gemini_client.py` 中，是模块级常量（`_VERIFY_*_PROMPT`、`_MERGE_PROMPT`）。

### 添加新的属性处理器
1. 实现 `AttributeHandlerPlugin` 协议（`handlers/plugins/protocol.py`）
2. 在 `di/container.py` 中注册
3. 在 `schemas/template_spec.py` 的 `HANDLER_BY_SCOPE` 中添加映射

### 启用真实 Gemini 调用
在 `config/.env` 中设置：
```ini
MVP_FORCE_GEMINI_MOCK=0
GEMINI_API_KEY=your_real_key
```

### 启用动态 Planner
```ini
MVP_DISABLE_PLANNER=0
```
