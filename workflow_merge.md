# Merge 模块定义

## 定位

Merge 是 Pipeline 的**最终裁决阶段**。它接收前面所有步骤（detect / nms / verify / quality / semantic / negative）的输出，对每个候选框做出**接受/拒绝判定**，并**跨候选框解决属性冲突**，最终产出一份干净的标注结果。

Merge 已被重构为**全确定性、零 LLM 调用**的组件。早期版本依赖 Gemini 读取执行日志来决策，现在完全由代码规则驱动。

---

## 整体数据流

```
PipelinePlan
  │
  ├─ detect ──→ nms ──→ verify ──→ quality ──→ semantic ──→ negative
  │                                                              │
  │                          Candidate[]                         │
  │   (每个候选框携带: verification, attributes, quality,         │
  │    negative_flags, 各阶段 confidence 分解)                    │
  │                                                              │
  ▼                                                              │
MergeEngine.merge()  ◄───────────────────────────────────────────┘
  │
  ├─ objects[]              → ObjectStateBuilder → AnnotationObject → annotation_result
  ├─ reasoning_trace[]      → RuntimeTrace.merge_reasoning
  ├─ resolved_attributes{}  → RuntimeTrace.resolved_attributes
  └─ merge_rules{}          → RuntimeTrace.meta.merge_rules
```

---

## 核心模块

### 1. MergeEngine (`runtime/merge_engine.py`)

确定性合并引擎，零外部依赖。

**初始化：**
```python
MergeEngine(merge_rules: dict | None = None)
```
- `merge_rules` 默认从 `config/merge_rules.json` 加载
- 核心参数：`weights`（detector/verifier 权重）、`attribute_confidence_threshold`

**入口方法：**
```python
def merge(
    *,
    image_path: str,
    parsed: ParsedTaskSpec,
    candidates_data: list[dict[str, Any]],  # Candidate.to_dict() 列表
    scene_pure_negative: bool = False,
) -> dict[str, Any]
```

**输出结构：**
```python
{
    "adapter": "MergeEngine",              # 标识来源
    "objects": [                           # 每个候选框一个 entry
        {
            "object_id": str,
            "is_positive": bool,           # 最终判定
            "negative_category": str | None,  # 被拒原因
            "confidence": float,           # 最终置信度
            "detection_confidence": float, # 分解保留
            "verification_confidence": float,
            "merge_confidence": float,
            "attributes": {
                attr_key: {"value": ..., "confidence": float}
            },
            "quality": {
                attr_key: {"value": ..., "confidence": float}
            },
            "negative_flags": {
                flag_key: {"value": bool, "confidence": float}
            }
        }
    ],
    "reasoning_trace": [                   # 审计追踪
        {"step": str, "input": str, "output": str, "reasoning": str},
        ...
    ],
    "resolved_attributes": {              # 跨候选框去重
        attr_key: {"value": ..., "confidence": float, "uncertain": bool}
    },
    "merge_rules": {...}                   # 回传配置
}
```

### 2. GeminiMerger (`models/gemini_merger.py`)

**薄封装层**，保持向后兼容。所有实际逻辑委托给 `MergeEngine`。

```python
class GeminiMerger:
    async def merge(self, ..., run_id="", execution_log_text="") -> dict:
        return self._engine.merge(...)   # 完全委托
```

- `tracer` 参数接受但不使用（merge 已无 LLM，无需 trace）
- `execution_log_text` 参数接受但不使用（旧版 Gemini merge 需看文本日志，现已不需要）

### 3. ObjectStateBuilder (`runtime/object_state_builder.py`)

将 `MergeEngine` 输出 + `Candidate` 状态拼接成两套视图：

**`build_annotation_object()` → `AnnotationObject`**（对外消费）：
- `status`: `"accepted"`（is_positive=true）/ `"rejected"`（is_positive=false）/ `"pending"`
- `attributes`: 只包含语义属性（非 null、非 infeasible），从 merge_panel 提取
- `confidence`: 以 merge_panel 为准

**`build()` → `ObjectState`**（内部全量状态）：
- 保留所有字段：verification, attributes, quality, negative_flags, 各阶段 confidence

**merge_panel 优先级规则：**
- `is_positive` 是最终权威 → 覆盖 pipeline 内的 negative/confidence
- 不存在 merge_panel 时用 Candidate 原始状态兜底

---

## 决策规则

### 正负判定（每条候选框独立）

⚠️ 自 v2.1 已从硬覆盖改为阈值判定：negative_flag 不再强制 override，而是通过置信度扣分参与决策。

| 条件 | is_positive | negative_category |
|------|-------------|-------------------|
| scene_pure_negative = true | 全部 negative | "pure_negative" |
| verify.ok = false | false | null |
| 任一 negative_flag.value = true | 由 merge_conf ± 惩罚后 > 0.3 决定 | 触发该 flag 的 name |
| 以上都不满足 | true | null |

**v2.1 变更**：`is_positive = verif_ok and merge_conf > 0.3`，negative_flag 对 merge_conf 扣 -0.10 而非直接置 negative。解决了原有 verify.ok 与 negative_flag 冲突时信息丢失的问题。

### 置信度计算

```
merge_confidence = detector_score * w_det + verifier_score * w_ver
                  + 调整值（见下方）
```

默认权重（`config/merge_rules.json`）：
- `w_det` = 0.3
- `w_ver` = 0.7

**调整规则：**

| 条件 | 调整 | 含义 |
|------|------|------|
| verify.ok AND det_score > 0.5 | +0.05 | 检测与验证一致 |
| verify.ok AND det_score ≤ 0.3 | -0.10 | 验证挽回了低分检测 |
| verify.ok=false AND det_score > 0.7 | -0.10 | 验证拒绝强检测 → 可疑 |
| 所有语义属性 confidence ≥ 0.7 | +0.05 | 属性一致性强 |
| 任一 negative_flag 触发 | -0.10 | 负样本惩罚 |

最终 `merge_confidence` clamp 到 [0.0, 1.0]。

### 属性冲突解决（跨候选框） — v2.1 新增 conflict trace

仅对 `is_positive = true` 的候选框做属性合并，**v2.1 增加完整投票追踪和分歧量化**：

```
for each attr_key:
    收集所有正候选框的投票（value + conf）
    取 confidence 最高的值作为 winner
    uncertain = confidence < attribute_confidence_threshold (默认 0.3)
    附加所有候选框的投票列表 → candidates: [{value, conf}, ...]
    计算冲突熵 → entropy (0.0=一致, >0 表示有分歧)
```

结果示例：
```python
resolved_attributes: {
    "package_form": {
        "value": "box",
        "confidence": 0.85,
        "uncertain": False,
        "candidates": [                    # ← v2.1 新增
            {"value": "box", "conf": 0.85},
            {"value": "bag", "conf": 0.82},
        ],
        "entropy": 0.998                  # ← v2.1 新增（接近 1bit = 二分分歧）
    }
}
```

### Pure Negative 短路

当 `scene_pure_negative = true` 时，MergeEngine 直接返回空 objects：
```python
return {"adapter": "MergeEngine", "objects": [], 
        "reasoning_trace": [scene_check_trace],
        "resolved_attributes": {},
        "merge_rules": self._rules}
```

---

## 配置（`config/merge_rules.json`）

```json
{
    "weights": {
        "detector": 0.3,
        "verifier": 0.7
    },
    "attribute_confidence_threshold": 0.3,
    "nms_iou_threshold": 0.5
}
```

| 参数 | 默认值 | 用途 |
|------|--------|------|
| `weights.detector` | 0.3 | 检测置信度权重 |
| `weights.verifier` | 0.7 | 验证置信度权重 |
| `attribute_confidence_threshold` | 0.3 | 低于此值标记 uncertain |
| `nms_iou_threshold` | 0.5 | NMS IoU 阈值（被 nms.py 使用） |

配置加载逻辑（`_load_merge_rules`）：
1. 尝试读取 `config/merge_rules.json`
2. 文件不存在或解析失败 → 返回硬编码默认值

---

## 与上下游的接口

### 上游：StepExecutor

在 `runtime/step_executor.py` 的 `_run_merge()` 中调用：

```python
async def _run_merge(self, step, ctx, state, image_path, parsed, run_id):
    merge_result = await self._merger.merge(
        image_path=image_path,
        parsed=parsed,
        candidates_data=[c.to_dict() for c in state.candidates],
        scene_pure_negative=state.scene_flags.get("pure_negative", False),
        run_id=run_id,
        execution_log_text="\n".join(ctx.execution_log_lines),
    )
    state._merge_result = merge_result
```

关键：`candidates_data` 是每个 `Candidate.to_dict()` 的结果，包含了该候选框从前序步骤积累的所有状态。

### 下游：RuntimeEngine

在 `runtime/engine.py` 的 `run_with_plan()` 中拆分为两条路径：

```python
merge_objects = result.merge_result.get("objects", [])
```

1. **AnnotationResult**（对外）：
   ```python
   panel = merge_objects[i] if i < len(merge_objects) else None
   ObjectStateBuilder.build_annotation_object(candidate, merge_panel=panel, ...)
   ```

2. **RuntimeTrace**（调试）：
   ```python
   trace.merge_reasoning  = result.merge_result.get("reasoning_trace", [])
   trace.resolved_attributes = result.merge_result.get("resolved_attributes", {})
   trace.meta.merge_adapter = result.merge_result.get("adapter", "unknown")
   trace.meta.merge_rules = result.merge_result.get("merge_rules", {})
   ```

---

## 特殊场景处理

### 1. 空候选框（零检测）
- merge 正常执行，`objects` 为空列表
- `reasoning_trace` 记录 YOLO 检测数 0
- `AnnotationResult.objects` 为空

### 2. Pure Negative 场景
- `scene_pure_negative = true` 时提前返回空 objects
- pipeline 内部由 `early_exit_rules` 机制保证 merge 前已退出

### 3. 属性缺失 / infeasible
- 缺失属性：`attributes` 中 key 不存在值
- Infeasible 属性：`attributes` 中 `value` = null 但 `infeasible` = true
- MergeEngine 不做特殊处理，仅按原始数据透传
- ObjectStateBuilder 在 `build_annotation_object()` 中过滤掉 infeasible 属性

### 4. 多候选框重叠
- NMS 在 merge 之前已通过 `apply_nms()` 抑制重叠候选框
- MergeEngine 对所有非 suppressed 候选框逐一处理

### 5. 异步批处理
- MergeEngine 独立处理每个帧，不存在跨帧状态依赖
- 批处理场景下每帧的 merge 是完全独立的

---

## 设计问题（已知待改）

### 问题 1：MergeEngine 职责过载

MergeEngine 在同一个函数中做了 **3 类不同层级** 的决策：

| 决策 | 本质类型 | 当前实现位置 |
|------|----------|-------------|
| `is_positive` / `negative_category` | **Classification Decision** | `MergeEngine.merge()` 内循环 |
| `merge_confidence` + 调整规则 | **Scoring Function** | `MergeEngine._weighted_confidence()` + 内联调整 |
| `resolved_attributes` | **Aggregation / Reconciliation** | `MergeEngine.merge()` 末尾 |

**建议**：拆成 3 个内部子模块（不一定拆文件）：

```
MergeEngine
 ├── DecisionPolicy       # is_positive / negative_category
 ├── ConfidencePolicy     # merge_confidence (weighted + adjustments)
 └── AttributeResolver    # resolved_attributes (含 conflict trace)
```

---

### ~~问题 2：negative_flag > verify 的隐性冲突~~ ✅ v2.1 已修复

**修复方案**：negative_flag 改为 penalty 而非 hard override。
- `is_positive = verif_ok and merge_conf > 0.3`（阈值判定）
- negative_flag 触发时 merge_conf 扣 -0.10（原有扣分逻辑保持不变）
- `neg_category` 保留用于追踪，不再影响判定结果

---

### ~~问题 3：NMS 与 Merge 的边界不一致~~ ✅ v2.1 已修复

**修复方案**：`StepExecutor._run_merge()` 中过滤 SUPPRESSED 候选框。
```python
candidates_data = [
    c.to_dict() for c in state.candidates
    if c.state is not CandidateState.SUPPRESSED
]
```

---

### ~~问题 4：resolved_attributes 缺少冲突语义~~ ✅ v2.1 已修复

**修复方案**：winner-takes-all 基础上增加完整投票列表和分歧量化指标。
- `candidates`: 所有正候选框的 [{value, conf}, ...] 投票列表
- `entropy`: 基于值分布的信息熵（0.0=完全一致，>0 表示有分歧）

---

### 问题 5：无 LLM 回退机制

MergeEngine 是纯代码规则，当规则覆盖不到的场景（如 verify 和 negative_flag 同时触发）时，没有 fallback 到 LLM 做二次裁决的路径。

---

## 与 NMS 的关系（当前实现）

NMS（`runtime/nms.py`）和 Merge 的当前边界：

| 阶段 | NMS | Merge |
|------|-----|-------|
| 输入 | YOLO 检测结果 | 全候选框（含 verify/quality/semantic/negative） |
| 目标 | 消除几何冗余（同一物理物体多个框） | 判定正负、合并属性、给出最终置信度 |
| 方法 | IoU 阈值（空间重叠过滤） | 加权投票 + 规则判定 |
| 状态变更 | `transition_to(SUPPRESSED)` | 写 `state._merge_result`（不修改候选框状态） |
| 依赖 | bbox 几何 + detector_score | verification / attributes / negative_flags 等全量数据 |

⚠️ **已知 gap**：MergeEngine 当前不区分 SUPPRESSED/ACTIVE，见问题 3。

---

## 测试覆盖

| 测试文件 | 测试类/函数 | 覆盖内容 |
|----------|------------|----------|
| `test_handler_output_formats.py` | `TestFullPipelineOutput.test_merge_output_format` | merge 输出的 panel 格式：object_id, is_positive, 各阶段 confidence |
| `test_handler_output_formats.py` | `TestWeightedVoting` | `_weighted_confidence` 默认/自定义权重 |
| `test_handler_output_formats.py` | `TestWeightedVoting.test_load_merge_rules_has_expected_keys` | merge_rules.json 读取 |
| `test_handler_output_formats.py` | `TestWeightedVoting.test_merge_output_contains_rules` | `meta.merge_rules` 传递 |
| `test_handler_output_formats.py` | `TestResolvedAttributes` | resolved_attributes 格式: value, confidence, uncertain；高置信度获胜逻辑 |
| `test_handler_output_formats.py` | `TestPipelineWithOverlap.test_overlapping_detections_merged` | 重叠候选框合并 |
| `test_handler_output_formats.py` | `TestNMS` | NMS 抑制记录、pipeline 中包含 nms step |
| `test_handler_output_formats.py` | `TestFullPipelineOutput` | 全链路 trace 中的 merge_reasoning 和 annotation_panel |

---

## 核心代码路径摘要

```
engine.run()
  └─ compile_plan()
  └─ executor.execute()
       └─ for each step in plan:
            _execute_one(step)
              └─ _dispatch()
                   ├─ "detect"   → _run_detect()
                   ├─ "nms"      → _run_nms()
                   ├─ "verify"   → _run_verify()
                   ├─ "quality"  → _run_quality()
                   ├─ "semantic" → _run_semantic()
                   ├─ "negative" → _run_negative()
                   └─ "merge"    → _run_merge()
                                   └─ GeminiMerger.merge()
                                        └─ MergeEngine.merge()
                                             ├─ scene_pure_negative → early return {}
                                             ├─ for each candidate:
                                             │    ├─ _weighted_confidence()
                                             │    ├─ adjustments
                                             │    └─ is_positive / neg_category
                                             ├─ resolve attributes
                                             └─ build reasoning_trace
  └─ engine.run_with_plan()
       └─ ObjectStateBuilder.build_annotation_object(merge_panel)
       └─ ObjectStateBuilder.build(merge_panel)
```
