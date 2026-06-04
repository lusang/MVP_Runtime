# MVP Runtime — Step Execution Workflow

> 版本：v1.0  
> 最后更新：2026-05-25

---

## 概述

本文档描述 Worker 拿到 PipelinePlan 后，StepExecutor 如何逐步骤执行、每一步内部做什么、数据如何在步骤之间流动。

```
AsyncWorker (batch level)
  │
  ├─ TemplateParser.parse(template)        → ParsedTaskSpec
  ├─ Planner.compile(parsed)               → PipelinePlan (batch 级，仅一次)
  │
  └─ for each task/frame:
       ├─ resolve image URL
       └─ engine.run_with_plan(image, plan, parsed)
            │
            └─ StepExecutor.execute(plan, image_path, parsed, run_id)
                 │
                 ├─ _ExecutionContext (steps 共享的可变上下文)
                 │
                 └─ for step in plan.steps (按 order 升序):
                      └─ _dispatch(step, ctx, ...)
                           ├─ "negative" (scene-level)   → _run_scene_negative
                           ├─ "detect"                   → _run_detect
                           ├─ "nms"                      → _run_nms
                           ├─ "verify"                   → _run_verify
                           ├─ "quality"                  → _run_quality
                           ├─ "attribute"/"semantic"     → _run_semantic
                           ├─ "negative" (per-candidate) → _run_negative
                           └─ "merge"                    → _run_merge
```

---

## 1. 入口：AsyncWorker

**文件：** `runtime/async_worker.py`

### 职责

Worker 是批处理（batch）级别入口，负责编排"一次解析 → 逐帧执行"的流程。

### 流程

```
process_batch(batch_id, tasks):
  │
  ├─ 1. 取第一个 task 的 template
  ├─ 2. TemplateParser.parse(template)     → ParsedTaskSpec
  ├─ 3. Planner.compile(parsed)            → PipelinePlan  (batch 级，一次)
  ├─ 4. 遍历 tasks (通常 1 个或多个 frame):
  │     └─ _process_task(task, plan, parsed):
  │          ├─ async_resolve_url(task.url) → 下载图片到 storage/origin
  │          ├─ engine.run_with_plan(plan, image_path, parsed, run_id)
  │          │    └─ StepExecutor.execute() → RuntimeResult
  │          ├─ engine.build_response()     → AnnotationResult + RuntimeTrace
  │          └─ callback(result)           → 回调通知外部
  │
  └─ 5. 返回 batch 结果列表
```

### 关键设计决策

- **Planner.compile() 在 batch 级调用一次**，同一 batch 内所有 task/frame 共享同一个 PipelinePlan
- 这意味着同一模板的所有 frame 执行相同的步骤拓扑，只是每帧检测/推理结果不同
- Worker 本身不参与 step 调度逻辑，它只负责"下载图片 → 执行 → 回调"

---

## 2. RuntimeEngine

**文件：** `runtime/engine.py`

### 职责

桥接 AsyncWorker 和 StepExecutor，同时负责组装最终返回结果。

### 流程

```python
class RuntimeEngine:
    def run_with_plan(self, plan, image_path, parsed, run_id):
        # 1. 创建 StepExecutor（注入 detector、verifier、merger 等）
        executor = StepExecutor(
            detector=self._detector,
            verifier=self._verifier,
            verification_handler=self._verification_handler,
            attribute_handler=self._attribute_handler,
            merger=self._merger,
            tracker=self._tracker,
        )

        # 2. 执行
        result = await executor.execute(
            plan=plan,
            image_path=image_path,
            parsed=parsed,
            run_id=run_id,
        )

        # 3. 组装响应
        annotation = AnnotationResult(
            object_name=plan.object_name,
            candidates=[...],      # 从 result.candidates 转换
            merge_result=result.merge_result,
            scene_pure_negative=result.scene_pure_negative,
        )

        trace = RuntimeTrace(
            plan_id=plan.plan_id,
            steps=[...],           # 从 result.executed_steps 转换
        )

        return annotation, trace
```

---

## 3. StepExecutor 核心

**文件：** `runtime/step_executor.py`

### 3.1 _ExecutionContext

所有 step 共享的可变上下文，是步骤间的数据通道。

```python
@dataclass
class _ExecutionContext:
    detections: list[Detection] = None        # detect 步骤的输出
    candidates: list[CandidateState] = None   # 所有候选框状态（从 detect 开始构建，后续各步不断更新）
    scene_pure_negative: bool = False         # scene-level negative 检测结果
    executed_step_ids: list[str] = None       # 已执行步骤 ID 列表
    execution_log_lines: list[str] = None     # 执行日志
    feasibility_rules: list = None            # 可行性规则（由 _init_context 中构建）
```

### 3.2 execute() 主循环

```python
async def execute(self, plan, image_path, parsed, run_id):
    ctx = self._init_context(plan, parsed)

    for step in plan.steps:
        # 1. 跳过条件检查
        if self._should_skip(step, ctx):
            continue

        # 2. 早退条件检查
        scene_neg = await self._check_early_exit(step, plan, ctx, image_path, parsed, run_id)
        if scene_neg:
            break

        # 3. 分发执行
        result = await self._dispatch(step, ctx, image_path, parsed, run_id, performance_tracker)

    # 4. 构建 RuntimeResult
    return self._build_result(ctx)
```

### 3.3 _dispatch() 路由

```python
async def _dispatch(self, step, ctx, image_path, parsed, run_id):
    if step.step == "detect":
        await self._run_detect(step, ctx, image_path, parsed, run_id)
    elif step.step == "nms":
        self._run_nms(step, ctx)
    elif step.step == "verify":
        await self._run_verify(step, ctx, image_path, parsed)
    elif step.step == "quality":
        await self._run_quality(step, ctx, image_path, parsed)
    elif step.step in ("attribute", "semantic"):
        await self._run_semantic(step, ctx, image_path, parsed)
    elif step.step == "negative":
        # 场景级 vs 逐候选级由 step 属性决定：
        #   data_flow=full_image, per_candidate=false → 场景级
        #   data_flow=crop,      per_candidate=true  → 逐候选级
        await self._run_negative(step, ctx, image_path, parsed)
    elif step.step == "merge":
        await self._run_merge(step, ctx, image_path, parsed, run_id)
```

---

## 4. 各 Step 详解

### 4.1 Scene-level Negative（step_type = "negative", 场景级）

**order = 0**，发生在 detect 之前。

#### 输入

| 来源 | 数据 |
|------|------|
| `plan.steps[0].params` | `attribute_key: "Pure Negative"` |
| 原始图片 | `image_path` |

#### 执行

```python
async def _run_scene_negative(self, step, ctx, image_path, parsed):
    # 调用 GeminiVerifier.verify_scene_pure_negative()
    # 输入：整张图片
    # 输出：{ "is_pure_negative": bool, "reason": str }

    result = await self._verifier.verify_scene_pure_negative(
        image_path=image_path,
        parsed=parsed,  # 含 negative prompt 模板
    )

    ctx.scene_pure_negative = result.get("is_pure_negative", False)
```

#### 输出

| 写入 ctx | 类型 |
|----------|------|
| `ctx.scene_pure_negative` | `bool` |

#### 后续影响

- **scene_pure_negative = true** → 触发 early_exit，跳过 detect 及之后所有步骤
- **scene_pure_negative = false** → 继续下一个 step

---

### 4.2 Detect

**order = 1**，scene-level negative 之后、NMS 之前。

#### 输入

| 来源 | 数据 |
|------|------|
| 原始图片 | `image_path` |
| `parsed.object_name` | 目标对象名称（如 "object"） |

#### 执行

```python
async def _run_detect(self, step, ctx, image_path, parsed, run_id):
    # 1. 调用 YOLODetector.detect()
    #    输入：整张图片 + object_name
    #    输出：list[Detection]

    detections = await self._detector.detect(
        image_path=image_path,
        target_object=parsed.object_name,
        parsed=parsed,
        run_id=run_id,
    )

    # 2. 保存检测结果
    ctx.detections = detections

    # 3. 构建 CandidateState 列表（每个检测框一个）
    for idx, det in enumerate(detections):
        bbox = det.bbox
        # 如果有 crop_path（抠图路径），analysis_bbox 自动设为 full crop
        analysis_path = det.crop_path or image_path
        analysis_bbox = bbox_for_full_crop(analysis_path) if det.crop_path else bbox

        c = CandidateState(
            object_id=f"obj_{idx}",
            detector_score=det.score,
            bbox=bbox,
            crop_path=det.crop_path,       # storage/preselection/{run_id}/candidate_{idx}.jpg
            analysis_path=analysis_path,
            analysis_bbox=analysis_bbox,
            exists=True,                    # 初始所有候选框都被标记为存在
        )
        ctx.candidates.append(c)
```

#### YOLODetector 内部细节

```python
class YOLODetector:
    async def detect(self, image_path, target_object, parsed, run_id):
        if os.environ.get("MVP_FORCE_YOLO_MOCK") == "1":
            return self._mock_detect()      # 返回固定检测结果（2 个候选框）

        # 真实模式：
        # 1. 解析 ObjectTarget（目标对象名称、置信度阈值等）
        # 2. 加载 YOLO World v2-X 模型
        # 3. inference 整图
        # 4. NMS 初步过滤（model-level，与 runtime 的 NMS step 不同）
        # 5. 对每个 bbox crop 出子图
        # 6. 保存到 storage/preselection/{run_id}/candidate_{idx}.jpg
        # 7. 返回 Detection(score, bbox, crop_path, ...)
```

#### 输出

| 写入 ctx | 类型 | 说明 |
|----------|------|------|
| `ctx.detections` | `list[Detection]` | 原始检测结果 |
| `ctx.candidates` | `list[CandidateState]` | 候选框列表，每个包含 bbox、crop_path、score |

---

### 4.3 NMS（Non-Maximum Suppression）

**order = 2**，detect 之后、verify 之前。

#### 输入

| 来源 | 数据 |
|------|------|
| `ctx.candidates` | 所有候选框（含 bbox 和 detector_score） |
| `step.params` | `{ iou_threshold: 0.5 }` |

#### 执行

```python
def _run_nms(self, step, ctx):
    # 调用 apply_nms() — 纯算法，无外部依赖
    # 算法：对 candidates 按 detector_score 降序排列，
    #       遍历，对每个保留的候选框，移除与其 IoU > threshold 的候选框

    iou_threshold = step.params.get("iou_threshold", 0.5)
    apply_nms(ctx.candidates, iou_threshold=iou_threshold)

    # 被抑制的候选框：c.exists = False
    # 未被抑制的候选框：c.exists = True（不变）
```

#### NMS 内部算法

```python
def apply_nms(candidates, iou_threshold=0.5):
    # 1. 按 detector_score 降序排列
    # 2. 遍历：对每个未被抑制的候选框 A
    #     遍历其后的所有候选框 B
    #       如果 A 与 B 的 IoU > threshold → 抑制 B（B.exists = False）
    # 3. 记录每个候选框的 analysis_history，添加 nms 决策记录
```

#### 输出

| 写入 ctx | 变更 |
|----------|------|
| `ctx.candidates[*].exists` | 被抑制的候选框设为 `False` |
| `ctx.candidates[*].analysis_history` | 追加 `{ step: "nms", decision: "kept"|"suppressed" }` |

---

### 4.4 Verify

**order = 3**，NMS 之后、quality 之前。

#### 输入

| 来源 | 数据 |
|------|------|
| `ctx.candidates`（仅 `exists=True` 的候选框） | 候选框 |
| `c.analysis_path` | 抠图路径（或原图路径） |
| `c.analysis_bbox` | 分析区域 bbox |
| `parsed` | 模板定义（含 verify prompt） |

#### 执行

```python
async def _run_verify(self, step, ctx, image_path, parsed):
    for c in ctx.candidates:
        if not c.exists:
            continue

        # 调用 VerificationHandler.verify_object()
        # → 内部调用 GeminiVerifier.verify_object()

        ver = await self._verification_handler.verify_object(
            image_path=c.analysis_path,   # 候选框抠图
            bbox=c.analysis_bbox,         # 分析区域
            parsed=parsed,                # 含 verify prompt
            object_id=c.object_id,
        )

        # 更新 candidate state
        c.verification = ver
        c.verify_score = float(ver.get("score", 0.0))
        c.compute_confidence()  # 综合 detector_score + verify_score

        # 如果验证未通过，标记为不存在
        if ver.get("ok") is False:
            c.exists = False
```

#### VerificationHandler 内部细节

```python
class VerificationHandler:
    async def verify_object(self, image_path, bbox, parsed, object_id):
        # 调用 GeminiVerifier.verify_object()
        # prompt 模板来自 parsed.object_name + verify_instruction
        # 输出：{ "ok": bool, "score": float, "rationale": str }
        return await self._verifier.verify_object(...)
```

#### GeminiVerifier 内部细节（Mock 模式）

```python
class GeminiVerifier:
    async def verify_object(self, ...):
        if os.environ.get("MVP_FORCE_GEMINI_MOCK") == "1":
            # 固定返回：{ ok: true, score: 0.88, rationale: "Mock verification passed" }
            return self._mock_response(...)
        # 真实模式：调用 GeminiClient API
```

#### 输出

| 写入 ctx | 类型 | 说明 |
|----------|------|------|
| `c.verification` | `dict` | 验证结果全量 |
| `c.verify_score` | `float` | 验证置信度 |
| `c.confidence` | `float` | 综合置信度（detector + verify 加权） |
| `c.exists` | `bool` | 验证失败 → `False`，后续步骤会跳过 |

---

### 4.5 Quality

**order = 4**，verify 之后、attribute 之前。

#### 输入

| 来源 | 数据 |
|------|------|
| `ctx.candidates`（仅 exists=True） | 候选框 |
| `c.analysis_path` / `c.analysis_bbox` | 分析区域 |
| `parsed.all_attribute_slots` | 所有属性定义 |
| `step.params.attribute_keys` | 需要执行的 quality 属性列表 |

#### 执行

```python
async def _run_quality(self, step, ctx, image_path, parsed):
    for c in ctx.candidates:
        if not c.exists:
            continue

        # 调用 AttributeHandler.analyze_by_scopes()
        result = await self._attribute_handler.analyze_by_scopes(
            image_path=c.analysis_path,
            bbox=c.analysis_bbox,
            parsed=parsed,
            object_id=c.object_id,
            scopes={"quality"},        # 只处理 scope=quality 的属性
            include_keys=frozenset(step.params.get("attribute_keys", [])),
        )

        # 更新 candidate state
        c.quality = result.quality
        # 将 quality 结果同步到 visibility
        c.visibility = dict(c.quality)

        # 如果 quality 分析包含数值指标（如模糊度、光照分），更新 metrics
        for kval, qitem in c.quality.items():
            if isinstance(qitem, dict) and "metrics" in qitem:
                c.metrics.update(qitem["metrics"])

        # 计算可行性（feasibility）
        _compute_feasibility(c, ctx.feasibility_rules)
```

#### AttributeHandler.analyze_by_scopes() 内部细节

```python
class AttributeHandler:
    async def analyze_by_scopes(self, image_path, bbox, parsed,
                                 object_id, scopes=None, skip_keys=None,
                                 include_keys=None, full_image_path=None, full_bbox=None):
        """
        遍历 parsed.all_attribute_slots，筛选符合条件的属性规格，
        对每个属性调用对应的 plugin 进行分析。
        """

        stage_result = AttributeStageResult()

        for spec in parsed.all_attribute_slots:
            # 1. 范围过滤
            if scopes and spec.scope not in scopes:
                continue
            # 2. 包含/排除键过滤
            if include_keys and spec.key not in include_keys:
                continue
            if skip_keys and spec.key in skip_keys:
                continue

            # 3. 从 registry 获取 plugin
            #    spec.handler → "gemini" / "gemini_negative" / "opencv_quality"
            plugin = self._registry.get(spec.handler)

            # 4. 调用 plugin.analyze()（详见下方各 plugin 章节）
            plugin_result = await plugin.analyze(
                image_path=image_path,
                bbox=bbox,
                spec=spec,
                full_image_path=full_image_path,
                full_bbox=full_bbox,
            )

            # 5. 根据 spec.scope 归入不同 bucket
            if spec.scope == "quality":
                stage_result.quality[spec.key] = plugin_result
            elif spec.scope == "semantic":
                stage_result.attributes[spec.key] = plugin_result
            elif spec.scope == "negative":
                stage_result.negative[spec.key] = plugin_result

        return stage_result
```

#### OpenCVQualityPlugin（quality scope 属性的 handler）

```python
class OpenCVQualityPlugin:
    async def analyze(self, image_path, bbox, spec, **kwargs):
        # 调用 OpenCVAnalyzer.analyze_quality()
        # 输入：抠图（或指定区域）
        # 输出：{ "value": str, "metrics": { "blur_score": 0.5, ... } }

        result = self._analyzer.analyze_quality(
            image_path=image_path,
            bbox=bbox,
            attribute_name=spec.key,   # 如 "blur" / "occlusion" / "lighting"
        )
        return result
```

#### OpenCVAnalyzer 内部细节

```python
class OpenCVAnalyzer:
    def analyze_quality(self, image_path, bbox, attribute_name):
        """
        属性名驱动的数值分析分发器。
        - blur:      Laplacian 方差（值越高越模糊）
        - occlusion: Canny 边缘密度
        - lighting:  直方图均值
        - background_clutter: 暂不支持数值分析 → 降级为 LLM 分析
        """

        if attribute_name == "blur":
            # Laplacian 方差
            score = ...   # 归一化后映射到 "none"/"low"/"medium"/"high"
            return { "value": level, "metrics": { "blur_score": score } }

        elif attribute_name == "occlusion":
            # Canny 边缘密度
            ...

        elif attribute_name == "background_clutter":
            # 当前 fixture 中 background_clutter 的 handler 是 "gemini" 而非 "opencv_quality"
            # 所以不会走到这里，而是走 GeminiAttributePlugin
            ...
```

#### 输出

| 写入 ctx | 类型 | 说明 |
|----------|------|------|
| `c.quality` | `dict[str, Any]` | 所有 quality 属性结果（keyed by 属性名） |
| `c.visibility` | `dict` | quality 结果的别名，用于后续步骤 |
| `c.metrics` | `dict` | 数值分析指标（如模糊度分数） |
| `c.missing_attributes` | `set` | feasibility 判定为不可行的属性列表 |

---

### 4.6 Attribute（Semantic）

**order = 5**，quality 之后、negative 之前。

#### 输入

| 来源 | 数据 |
|------|------|
| `ctx.candidates`（仅 exists=True） | 候选框 |
| `c.analysis_path` / `c.analysis_bbox` | 分析区域 |
| `parsed.all_attribute_slots` | 所有属性定义 |
| `step.params.attribute_keys` | 需要执行的语义属性列表 |
| `c.missing_attributes` | 需要跳过的不可行属性 |

#### 执行

```python
async def _run_semantic(self, step, ctx, image_path, parsed):
    # 判断使用全图还是抠图
    is_full = step.data_flow.value == "full_image"

    for c in ctx.candidates:
        if not c.exists:
            continue

        # 跳过不可行属性
        skip_keys = frozenset(c.missing_attributes) if c.missing_attributes else None

        # 根据 data_flow 选择分析区域
        effective_img = c.analysis_path if not is_full else image_path
        effective_bb  = c.analysis_bbox if not is_full else c.bbox

        result = await self._attribute_handler.analyze_by_scopes(
            image_path=effective_img,
            bbox=effective_bb,
            parsed=parsed,
            object_id=c.object_id,
            scopes={"semantic"},       # 只处理 scope=semantic 的属性
            skip_keys=skip_keys,
            include_keys=frozenset(step.params.get("attribute_keys", [])),
        )

        c.attributes.update(result.attributes)

        # 为不可行属性填充空值
        for key in c.missing_attributes:
            c.attributes[key] = {"value": None, "confidence": 0.0, "infeasible": True}
```

#### GeminiAttributePlugin（semantic scope 属性的 handler）

```python
class GeminiAttributePlugin:
    async def analyze(self, image_path, bbox, spec, **kwargs):
        # spec.scope == "semantic" 要求
        # 调用 GeminiVerifier.verify_attribute()
        # 输入：抠图 / 指定区域 + 属性描述 prompt
        # 输出：{ "value": any, "confidence": float, "rationale": str }

        result = await self._verifier.verify_attribute(
            image_path=image_path,
            bbox=bbox,
            attribute_spec=spec,   # 含 name, type, description, options
        )
        return result
```

#### 输出

| 写入 ctx | 类型 | 说明 |
|----------|------|------|
| `c.attributes` | `dict[str, dict]` | 语义属性分析结果 `{ "value": ..., "confidence": ... }` |

---

### 4.7 Negative（Per-candidate）

**order = 6**，attribute 之后、merge 之前。

#### 输入

| 来源 | 数据 |
|------|------|
| `ctx.candidates`（仅 exists=True） | 候选框 |
| `c.analysis_path` / `c.analysis_bbox` | 抠图区域 |
| `parsed.all_attribute_slots` | 所有属性定义 |
| `step.params.attribute_keys` | 需要执行的 negative 属性列表 |
| *额外参数* | `full_image_path` + `full_bbox`（全图上下文） |

#### 执行

```python
async def _run_negative(self, step, ctx, image_path, parsed):
    for c in ctx.candidates:
        if not c.exists:
            continue

        result = await self._attribute_handler.analyze_by_scopes(
            image_path=c.analysis_path,    # 主要分析区域：抠图
            bbox=c.analysis_bbox,
            parsed=parsed,
            object_id=c.object_id,
            scopes={"negative"},           # 只处理 scope=negative 的属性
            # 传入全图路径和全图 bbox 供 plugin 参考
            full_image_path=image_path,
            full_bbox=c.bbox,
        )

        c.negative_flags.update(result.negative)
```

#### GeminiNegativePlugin（negative scope 属性的 handler）

```python
class GeminiNegativePlugin:
    async def analyze(self, image_path, bbox, spec, full_image_path=None, full_bbox=None, **kwargs):
        # spec.scope == "negative" 要求
        # 调用 GeminiVerifier.verify_attribute()
        # 输入：抠图 + 整图（作为上下文）
        # 输出：{ "value": bool, "confidence": float, "rationale": str }

        result = await self._verifier.verify_attribute(
            image_path=image_path,         # 抠图路径
            bbox=bbox,
            attribute_spec=spec,           # 含 negative 判定 prompt
            full_image_path=full_image_path, # 整图（各作上下文）
            full_bbox=full_bbox,
        )
        return result
```

#### 输出

| 写入 ctx | 类型 | 说明 |
|----------|------|------|
| `c.negative_flags` | `dict[str, bool]` | 阴性判定结果（如 `ambiguous: false`） |

---

### 4.8 Merge

**order = 7**，最后一个 step。

#### 输入

| 来源 | 数据 |
|------|------|
| `ctx.candidates`（所有候选框） | 最终候选框状态 |
| `ctx.scene_pure_negative` | 场景级 negative 结果 |
| 原始图片 | `image_path` |
| `parsed` | 模板定义 |

#### 执行

```python
async def _run_merge(self, step, ctx, image_path, parsed, run_id):
    # 1. 将 CandidateState 转换为可序列化 dict
    candidates_data = [c.to_dict() for c in ctx.candidates]

    # 2. 调用 GeminiMerger.merge()
    merge_result = await self._merger.merge(
        image_path=image_path,
        parsed=parsed,
        candidates_data=candidates_data,
        scene_pure_negative=ctx.scene_pure_negative,
        run_id=run_id,
        execution_log_text="\n".join(ctx.execution_log_lines),
    )

    # 3. 保存 merge 结果
    ctx.merge_result = merge_result
```

#### GeminiMerger 内部细节

```python
class GeminiMerger:
    async def merge(self, image_path, parsed, candidates_data,
                     scene_pure_negative, run_id, execution_log_text):

        if os.environ.get("MVP_FORCE_GEMINI_MOCK") == "1":
            return self._mock_merge(candidates_data, scene_pure_negative)

        # 真实模式：调用 GeminiClient API
        # prompt 包含：所有候选框的检测/验证/属性/negative 结果
        # 输出：
        #   {
        #     "objects": [
        #       { "object_id": "obj_0", "is_positive": true,
        #         "merge_confidence": 0.989,
        #         "attributes": { ... } }
        #     ],
        #     "resolved_attributes": {
        #       "object_type": { "value": ["瓦楞纸箱"], "confidence": 0.85, "uncertain": false }
        #     },
        #     "adapter": "...",
        #     "merge_rules": [...],
        #   }
```

#### Mock 模式合并逻辑

```
merge_rules:
  - weighted_voting: detector=0.3, verifier=0.7
  - merge_agreement_bonus: +0.2 if 2+ candidates agree
  - merge_conflict_penalty: -0.2 if disagreement
  - attribute_conflict_resolution_by_max_confidence

对每个 positive candidate:
  merge_confidence = detector_score*0.3 + verify_score*0.7
  + 如果有 2+ candidates agree → merge_confidence += 0.2
  + 如果有 disagreement → merge_confidence -= 0.2

resolved_attributes:
  对候选框之间的属性取最高 confidence 的值
  如果最大值 < threshold → uncertain = true
```

#### 输出

| 写入 ctx | 类型 | 说明 |
|----------|------|------|
| `ctx.merge_result` | `dict` | 全量合并结果 |

---

## 5. 步骤间数据流总览

```
ExecutionContext
  │
  ├── detections: list[Detection]        ←── detect
  │
  ├── candidates: list[CandidateState]
  │     ├── bbox / crop_path             ←── detect
  │     ├── exists                       ←── detect → nms → verify
  │     ├── detector_score               ←── detect
  │     ├── verify_score                 ←── verify
  │     ├── confidence                   ←── verify (综合 score)
  │     ├── quality: dict                ←── quality
  │     ├── visibility: dict             ←── quality (alias)
  │     ├── metrics: dict                ←── quality
  │     ├── missing_attributes: set      ←── quality (feasibility)
  │     ├── attributes: dict             ←── attribute
  │     ├── negative_flags: dict         ←── negative
  │     └── analysis_history: list       ←── 所有步骤
  │
  ├── scene_pure_negative: bool          ←── scene negative
  │
  └── merge_result: dict                 ←── merge
```

---

## 6. Plugin/Handler 注册与调度

### 注册时机（应用启动时）

```python
registry = AttributeHandlerRegistry()
registry.register("gemini",           lambda: GeminiAttributePlugin(verifier))
registry.register("gemini_negative",  lambda: GeminiNegativePlugin(verifier))
registry.register("opencv_quality",   lambda: OpenCVQualityPlugin(analyzer))
```

### 调度逻辑

PipelinePlan 中每个 step 不直接指定 handler，而是：

```
PipelinePlan.step.params.attribute_keys → all_attribute_slots
  → 每个 slot 的 spec.handler → registry.get(spec.handler)
  → plugin.analyze(image, bbox, spec, ...)
```

这意味着 **AttributeHandler 内部遍历 all_attribute_slots**，按 spec.handler 分派到不同 plugin，而不是 StepExecutor 逐属性调度。

### Handler ↔ Scope ↔ Plugin 对照表

| Scope | Handler | Plugin | Model |
|-------|---------|--------|-------|
| `quality` | `opencv_quality` | `OpenCVQualityPlugin` | `OpenCVAnalyzer`（数值规则引擎） |
| `quality` | `gemini` | `GeminiAttributePlugin` | `GeminiVerifier` → Gemini LLM |
| `semantic` | `gemini` | `GeminiAttributePlugin` | `GeminiVerifier` → Gemini LLM |
| `negative` | `gemini_negative` | `GeminiNegativePlugin` | `GeminiVerifier` → Gemini LLM |

---

## 7. 跳过条件与早退

### 跳过条件（Skip Conditions）

在每个 step 执行前检查。跳过意味着 step 不执行，但流程继续。

| 条件 | 触发时机 | 效果 |
|------|---------|------|
| `scene_pure_negative` | 场景级 negative 触发后 | 跳过 per-candidate negative 步骤（已无需逐候选判断） |
| `all(not c.exists for c in candidates)` | 所有候选框被 NMS/verify 淘汰 | 跳过 quality / attribute / negative 步骤 |

### 早退条件（Early Exit）

早退意味着整个 pipeline 终止，后续所有 step 不执行。

| 条件 | 触发时机 | 效果 |
|------|---------|------|
| `scene_pure_negative = true` | scene negative 步骤后 | 跳过 detect → ... → merge 全部步骤 |

---

## 8. 超时与跟踪

### PerformanceTracker

每个 step 执行前后记录：

```python
tracker.record_step_start(plan_id, step.order, step.step, step.model_id)
# ... step execution ...
tracker.record_step_end(plan_id, step.order, step.step, status="success")
```

### 日志

```python
ctx.execution_log_lines.append(f"[{step.order}] {step.step} ({step.model_id}) → OK")
```

最终传递给 `GeminiMerger.merge()` 的 `execution_log_text`，供 LLM 了解执行历史。
