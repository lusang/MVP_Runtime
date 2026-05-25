# MVP Runtime — 执行计划

> 版本：v1.0  
> 日期：2026-05-25  
> 状态：待开始

---

## Phase 1 — 基础设施验证（无 AI 依赖）

**目标：** 不依赖任何 AI 模型，确认 Pipeline 骨架各环节输入输出正确。

### 1.1 验证 JSON/URL 输入输出

- [ ] 编写测试用例，覆盖 `POST /run_annotation_async` 的以下场景：
  - 合法请求 → 202 Accepted（已有）
  - 空 tasks → 400 Bad Request（已有）
  - 缺少 callback_url → 400 Bad Request
  - file:// 指向不存在文件 → 回调返回 failed 状态
  - http:// 下载正常文件 → 回调返回 completed 状态
  - http:// 下载超时 → 正确异常处理
- [ ] 验证回调重试机制：启动测试 HTTP server 接收回调，确认 3 次重试 + 死信队列写入

### 1.2 验证每一步 handler 输出格式

**当前状态：** `MVP_FORCE_GEMINI_MOCK=1`，所有 Gemini 步骤返回 mock 数据。

- [ ] 对每个 pipeline step，断言输出字段完整：
  - **detect** → `DetectionCandidate[]`（bbox, score, crop_path）
  - **verify** → `{ok, score, rationale}`
  - **quality** → `{value, confidence, metrics}`
  - **attribute** → `{value, confidence}`
  - **negative** → `{value, confidence}`
  - **merge** → `{objects[], reasoning_trace[]}`
- [ ] 验证边界情况：
  - 图片中无目标 → 0 个候选框，pipeline 正确执行
  - bbox 在图片边界外 → 裁剪异常时 mock fallback
  - 模板中无任何属性 → 跳过 attribute/quality/negative 步骤

---

## Phase 2 — 架构准备 ✅

**目标：** 为后续精细化和动态 Planner 改造打好基础。

### 2.1 TemplateAttributeSpec 增加 `analysis_scope` 字段 ✅

**改动位置：** `schemas/template_spec.py` — 增加 `analysis_scope: Literal["crop", "full_image"]` 字段，默认 `"crop"`。

**改动位置：** `runtime/template_parser.py` — `_parse_attribute_list` 解析 `analysis_scope`，非法值静默回退为 `"crop"`。

**向后兼容：** 无 `analysis_scope` 的旧模板全部默认 `"crop"`，原有行为不变。

### 2.2 提示词从常量抽成可配置模板 ✅

**新增文件：**
- `prompts/verify_object.txt`
- `prompts/verify_attribute.txt`
- `prompts/verify_negative_attribute.txt`
- `prompts/verify_scene_negative.txt`
- `prompts/merge.txt`
- `prompts/planner.txt`
- `runtime/prompt_manager.py` — `PromptManager` 类

**加载优先级：**
1. 环境变量 `MVP_PROMPT_{NAME}`（大写，- 转为 _）
2. `prompts/{name}.txt` 文件
3. Python 常量（硬编码 default fallback）

**改动的文件：** `models/gemini_client.py`, `runtime/planner.py` — 全部通过 `PromptManager.load()` 加载

### 2.3 Planner 按 `analysis_scope` 分组属性到不同 step ✅

**`_StaticPlanFactory.build()`：** semantic 属性按 `analysis_scope` 拆分：
- `crop` 属性 → `data_flow: CROP` 的 attribute step
- `full_image` 属性 → `data_flow: FULL` 的 attribute step
- 同 scope 属性合并到一个 step；两种 scope 都有则创建两个 step（crop 在前）

**`StepExecutor._run_semantic()`：** 根据 `step.data_flow` 路由：
- `CROP` → 使用 `c.analysis_path` / `c.analysis_bbox`（YOLO 裁剪图）
- `FULL` → 使用原始 `image_path` / `c.bbox`（原图 + 原框）
- 通过 `step.params["attribute_keys"]` 传给 `AttributeHandler` 按需过滤

**`AttributeHandler.analyze_by_scopes()`：** 新增 `include_keys` 参数过滤属性

---

## Phase 3 — 真实调用 ✅

**目标：** 连接真实模型，获取有意义的 trace 数据用于调优。

### 3.1 启用真实 Gemini 调用 ✅

- [x] 配置真实 `GEMINI_API_KEY`（已在 `.env` 中配置）
- [x] API 错误重试 / fallback 逻辑：
  - `models/gemini_client.py` — `_gemini_generate` 增加 `@backoff.on_exception` 重试
  - 重试条件：ServerError(5xx)、429(rate limit)、ConnectionError、TimeoutError
  - 策略：exponential backoff, max 3 次, max 60s
  - 客户端缓存：`_get_genai_client()` 缓存 `genai.Client` 实例，避免每次调用重建
- [x] 测试覆盖：client cache、retryable/non-retryable exception 判断
- [ ] 配置 `MVP_FORCE_GEMINI_MOCK=0` 后运行 pipeline 验证（需用户在 DataHub 端触发）

### 3.2 接入 LangFuse ✅

- [x] 安装 `langfuse` Python SDK
- [x] 创建 `runtime/tracer.py` — `GeminiTracer` 类
  - 可选集成：仅 `LANGFUSE_SECRET_KEY` + `LANGFUSE_PUBLIC_KEY` 配置后才启用
  - 未配置时完全 no-op（不抛异常、不产生网络请求）
  - 每个 `run_id` 创建一个 LangFuse trace
  - 每次 Gemini 调用通过 `tracer.observe()` 记录为 span
- [x] 接入 `GeminiClient` ：所有 4 个 API 方法（verify_object、verify_attribute、verify_scene_pure_negative、generate_merge）都调用 `self._trace()`
- [x] DI 容器注入：`build_container(tracer=...)` — 默认创建 `GeminiTracer()`（auto no-op）

### 3.3 根据 Trace 调优提示词

**前置条件：**
- [ ] DataHub 端配置 `LANGFUSE_SECRET_KEY` + `LANGFUSE_PUBLIC_KEY` 环境变量
- [ ] `MVP_FORCE_GEMINI_MOCK=0`，使用真实 Gemini 调用
- [ ] 收集至少 20 张测试图的执行结果
- [ ] 分析 LangFuse trace 中每步 Gemini 的输出质量
- [ ] 修改 `prompts/*.txt` 中的提示词，重新运行对比
- [ ] 建立简单的对比矩阵（每次改动前后指标）

---

## Phase 4 — Merge 增强 ✅

**目标：** 当一张图出现多个候选框时，merge 层能做出更合理的决策。

### 4.1 NMS 去重（确定性规则） ✅

- [x] 在 merge 前插入 NMS（Non-Maximum Suppression）步骤
- [x] 配置 IoU 阈值（如 0.5），高于阈值的重叠框只保留置信度最高的一个
- [x] 被抑制的候选框标记为 `rejected` 而非删除（保留审计轨迹）

**改动位置：**
- `data/bbox.py` — 新增 `BBox.area()` 方法和 `compute_iou()` 函数
- `runtime/nms.py` — `apply_nms()`: 按 detector_score 排序，N 方比较 IoU > threshold 时抑制低分框
- `runtime/step_executor.py` — 新增 `_run_nms` 方法，在 `_dispatch` 中路由；`_run_verify` 跳过已抑制候选
- `runtime/planner.py` — 静态计划在 detect 后、verify 前插入 NMS 步骤（model_id=rule-engine, iou_threshold=0.5）

### 4.2 加权投票规则 ✅

- [x] 配置 detector 和 verifier 的权重（如 detect:verify = 3:7）
- [x] 当两者冲突时（detect 高分 but verify rejected），按权重决策
- [x] 合并表决策略可配置（JSON 配置文件）

**改动位置：**
- `config/merge_rules.json` — 新增权重配置（detector=0.3, verifier=0.7）
- `models/gemini_merger.py` — 新增 `_load_merge_rules()` 和 `_compute_weighted_confidence()`；`_mock_merge()` 改用加权计算 merge_conf

### 4.3 属性冲突解决 ✅

- [x] 同一属性多个候选框都给出了有效值 → 取置信度最高的
- [x] 属性值置信度过低（< threshold）→ 标记为 `uncertain` 而非丢弃

**改动位置：**
- `models/gemini_merger.py` — `_mock_merge()` 新增跨候选框属性冲突解决逻辑，结果写入 `resolved_attributes`
- `data/io.py` — `RuntimeTrace` 新增 `resolved_attributes` 字段
- `runtime/engine.py` — trace 中透传 `resolved_attributes` 和 `merge_rules` 元数据

---

## Phase 5 — 模型扩展（远期）

**目标：** 扩展模型目录，支持更复杂的检测场景。

### 5.1 模型目录

- [ ] 注册人检测模型（YOLO 人检测）
- [ ] 注册行为分类模型或 prompt 模板
- [ ] 在 `runtime/model_registry.py` 中注册新模型

### 5.2 动态 Planner 路由

- [ ] 当模板中有要求全图判断的属性时，Planner 自动插入独立的属性 step
- [ ] 当模板中有"人的行为"类属性时，Planner 路由到人检测 + 行为分类 pipeline
- [ ] 动态 Planner 的决策逻辑从 Gemini 回退到规则引擎

---

## 优先级矩阵

| 序号 | 任务 | 复杂度 | 依赖 | 价值 | 优先级 |
|------|------|--------|------|------|--------|
| 1.1 | 验证 JSON/URL | 低 | 无 | 高 | P0 |
| 1.2 | 验证 handler 输出 | 低 | 无 | 高 | P0 ✅ |
| 2.1 | analysis_scope 字段 | 低 | DataHub 侧配合 | 高 | P0 ✅ |
| 2.2 | 提示词配置化 | 中 | 无 | 中 | P1 ✅ |
| 2.3 | Planner 按 scope 分组 | 中 | 2.1 | 高 | P0 ✅ |
| 3.1 | 启用真实 Gemini | 低 | 无 | 高 | P1 ✅ |
| 3.2 | 接入 LangFuse | 中 | 3.1 | 中 | P1 ✅ |
| 3.3 | 调优提示词 | 高 | 3.1, 2.2 | 高 | P1 |
| 4.1 | NMS 去重 | 中 | 无 | 中 | P2 ✅ |
| 4.2 | 加权投票 | 中 | 无 | 中 | P2 ✅ |
| 4.3 | 属性冲突解决 | 中 | 4.2 | 中 | P2 ✅ |
| 5.x | 模型扩展 | 高 | 2.3, 4.x | 低 | P3 |

---

## 立即开始的第一个 Issue

**Issue #1: 给模板属性增加 `analysis_scope` 字段**

1. 在 `schemas/template_spec.py` 的 `TemplateAttributeSpec` 中增加 `analysis_scope` 字段
2. 在 `runtime/template_parser.py` 的 `_parse_attribute_list` 中解析此字段
3. 通知 DataHub 侧在模板中输出此字段
4. 修改 `runtime/planner.py` 测试新分组逻辑
5. 修改 `runtime/step_executor.py` 支持按 data_flow 路由属性步骤

**预计工作量：** 2-4 小时
