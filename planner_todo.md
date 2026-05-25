# Planner Implementation — TODO

> 版本：v1.0 → v1.1 (updated 2026-05-25)
> 进度：Phase 1 完成 ✅ / Phase 2 完成 ✅ / Phase 3 待开始

---

## 架构总览

```
Template JSON
  → TemplateParser.parse()
    → ParsedTaskSpec
      │
      ▼
Stage 1: Semantic Classifier ───── runtime/semantic_classifier.py ✅
  输入:  attribute {name, type, description, scope}
  出力:  SemanticFeatures (feature vector)
  方式:  keyword rules (Phase 1) → LLM (Phase 2)
      │
      ▼
Stage 2a: Capability Mapping ───── runtime/capability_mapping.py ✅
  输入:  SemanticFeatures + scope
  出力:  {data_flow, required_capabilities, per_candidate}
      │
      ▼
Stage 2b: Resolver ─────────────── runtime/resolver.py ✅
  输入:  required_capabilities + scope + attribute
  出力:  {handler, model_id}
      │
      ▼
Stage 3: StepGraph Builder ─────── runtime/step_graph_builder.py ✅
  输入:  per-attribute runtime params
  出力:  PipelinePlan
      │
      ▼
Stage 4: Validator ─────────────── runtime/plan_validator.py ✅
  输入:  PipelinePlan
  出力:  校验通过 / 失败
      │
      ▼
  Planner.compile() ────────────── runtime/planner.py ✅ (重构)
  编排以上 4 个 stage
```

---

## Phase 1 — 新建语义层 ✅

### 1.1 SemanticFeatures 数据结构 ✅

**文件：** `schemas/semantic_features.py`

```python
@dataclass
class SemanticFeatures:
    needs_global_context: bool = True
    requires_reasoning: bool = True
    candidate_level: bool = True
    supports_numeric_analysis: bool = False
    requires_spatial_relation: bool = False
    requires_temporal_context: bool = False
    semantic_type: str = "unknown"
    reason: str = ""

@dataclass
class AttributeCapabilities:
    attribute_key: str
    data_flow: Literal["crop", "full_image"]
    required_capabilities: list[str]
    per_candidate: bool

@dataclass
class AttributeRuntimeParams:
    attribute_key: str
    data_flow, handler, per_candidate, model_id, required_capabilities, scope, prompt_key
```

### 1.2 Keyword-rule Semantic Classifier ✅

**文件：** `runtime/semantic_classifier.py`

- `classify_by_keywords(attr) -> SemanticFeatures`
- 关键词规则：name/description/type → feature vector
- NEVER uses `semantic_type` in conditionals
- 14 个测试覆盖所有 scope 类型

### 1.3 Capability Mapping ✅

**文件：** `runtime/capability_mapping.py`

- `map_features(features, attribute_key, scope) -> AttributeCapabilities`
- 纯确定性映射：feature vector → data_flow + capabilities
- 7 个测试

### 1.4 Resolver ✅

**文件：** `runtime/resolver.py`

- `resolve(caps, scope, attribute_name) -> AttributeRuntimeParams`
- 全部硬编码映射表
- High-stakes 属性 (is_package) → gemini-2.5-pro
- 9 个测试

### 1.5 StepGraph Builder ✅

**文件：** `runtime/step_graph_builder.py`

- `StepGraphBuilder.build(parsed, attribute_params) -> PipelinePlan`
- 固定拓扑：scene_neg → detect → nms → verify → quality → semantic → negative → merge
- 属性合并：groupby(scope, data_flow, handler)
- pure_negative 场景级 pre-check + early_exit
- 6 个测试

### 1.6 Validator ✅

**文件：** `runtime/plan_validator.py`

- `validate(plan) -> ValidationResult`
- FAIL: 缺 detect/merge, plan_id/object_name 为空, detect 在 merge 后
- WARN: quality 在 semantic 后, merge 不是最后, nms 位置不对
- 11 个测试

---

## Phase 2 — 集成 Compiler ✅

### 2.1 重构 Planner.compile() ✅

**文件：** `runtime/planner.py`

- `compile_plan(parsed)` → 4-stage pipeline
- `Planner.compile()` → 调用 compile_plan
- 保留 `_StaticPlanFactory` 作为 fallback
- 8 个集成测试 + trace 验证通过

### 2.2 HANDLER_BY_SCOPE ✅

- 标记为向后兼容保留
- `_StaticPlanFactory` 不再使用

### 2.3 DI 容器 ✅

- 规划器中不需要 DI（纯函数调用）
- StepExecutor 仍然通过 DI 注入

---

## Phase 3 — 集成测试与验证

### 3.1 集成测试 ✅

**文件：** `tests/test_planner_compiler.py`

- 8 个测试覆盖 fixture 和 Template.json
- 验证：steps 数量、early_exit、skip_conditions、属性合并、模型选择

### 3.2 验证 trace_plan.py ✅

**文件：** `temp/trace_plan.py`（更新）

```
=== Compiled PipelinePlan ===
  steps (8):
    [0] negative      flow=full_image  per_candidate=False  scene_check
    [1] detect        flow=full_image  per_candidate=False
    [2] nms           flow=full_image  per_candidate=False
    [3] verify        flow=crop        per_candidate=True
    [4] quality       flow=full_image  per_candidate=False  [background_clutter]
    [5] attribute     flow=crop        per_candidate=True   [object_type, is_package]
    [6] negative      flow=crop        per_candidate=True   [ambiguous, open_set_negative]
    [7] merge         flow=full_image  per_candidate=False
  early_exit: scene_pure_negative
  skip:       scene_pure_negative, all candidates rejected
```

### 3.3 待优化项

- [ ] 验证 `_StaticPlanFactory` fallback 路径在 `MVP_DISABLE_PLANNER=1` 时正常工作
- [ ] 旧测试 `test_engine_pipeline.py` 使用 DI 容器，需要确认 compile_plan 替换后无问题
- [ ] 考虑将 `_has_pure_negative` 提取为共享工具函数（现在 planner.py 和 step_graph_builder.py 各有一份）

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `schemas/semantic_features.py` | ✅ 新建 | SemanticFeatures + AttributeCapabilities + AttributeRuntimeParams |
| `runtime/semantic_classifier.py` | ✅ 新建 | Stage 1: 关键词规则分类器 |
| `runtime/capability_mapping.py` | ✅ 新建 | Stage 2a: feature → capabilities |
| `runtime/resolver.py` | ✅ 新建 | Stage 2b: capabilities → handler |
| `runtime/step_graph_builder.py` | ✅ 新建 | Stage 3: 构建 PipelinePlan |
| `runtime/plan_validator.py` | ✅ 新建 | Stage 4: 校验 PipelinePlan |
| `runtime/planner.py` | ✅ 重构 | compile() 编排 4-stage |
| `schemas/template_spec.py` | ⟡ 微调 | HANDLER_BY_SCOPE 保留向后兼容 |
| `temp/trace_plan.py` | ✅ 更新 | 使用 compile_plan |
| `tests/test_semantic_classifier.py` | ✅ 新建 | 14 tests |
| `tests/test_capability_mapping.py` | ✅ 新建 | 7 tests |
| `tests/test_resolver.py` | ✅ 新建 | 9 tests |
| `tests/test_step_graph_builder.py` | ✅ 新建 | 6 tests |
| `tests/test_plan_validator.py` | ✅ 新建 | 11 tests |
| `tests/test_planner_compiler.py` | ✅ 新建 | 8 tests |
| `planner_todo.md` | ✅ 更新 | 本文件 |

---

## 测试统计

```
59 passed (全部测试，包括原有)
  - 14  semantic_classifier
  - 7   capability_mapping
  - 9   resolver
  - 6   step_graph_builder
  - 11  plan_validator
  - 8   planner_compiler (integration)
  - 4   原有测试 (engine, parser, vocab, yolo)
```
