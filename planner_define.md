# MVP Runtime — Planner Architecture

> 版本：v2.1  
> 最后更新：2026-05-25

---

## 核心思想

Planner 不是一个 Agent，而是一个 **Compiler**。

```
┌─────────────────────────────────────────────────────────────┐
│                    Planner (Compiler)                       │
│                                                             │
│  Stage 1: Semantic Classifier (LLM)                         │
│    输入:  attribute {name, type, description, options}      │
│    输出:  semantic feature {needs_global_context, ...}     │
│                                                             │
│  Stage 2a: Capability Mapping (代码)                        │
│    输入:  semantic feature                                  │
│    输出:  {data_flow, required_capabilities, per_candidate} │
│                                                             │
│  Stage 2b: Resolver (代码, 全部硬编码)                       │
│    输入:  required_capabilities + scope                     │
│    输出:  {handler, model_id}                              │
│                                                             │
│  Stage 3: StepGraph Builder (代码)                          │
│    输入:  属性级 runtime params                             │
│    输出:  PipelinePlan (steps, ordering, grouping)          │
│                                                             │
│  Stage 4: Validator (代码)                                  │
│    输入:  PipelinePlan                                      │
│    输出:  校验通过 / 失败                                    │
└─────────────────────────────────────────────────────────────┘
```

### 为什么要分层？

| 旧架构（Planner = Agent） | 新架构（Planner = Compiler） |
|--------------------------|-----------------------------|
| LLM 输出完整的 PipelinePlan | LLM 只输出属性的语义特征 |
| 一次调用决定全部流程 | 分层解耦，每层独立可测 |
| Runtime 逻辑与语义推理混合 | Runtime 逻辑完全 deterministic |
| Prompt 中需要描述 plan schema / topology / validator | Prompt 中只有属性本身 |
| 难以测试 | 每层可单独写 unittest |
| 状态在 prompt 中间层爆炸 | 依赖单向传递，无耦合 |

---

## Stage 1：Semantic Classifier（LLM）

### 职责

LLM 在这里只做它擅长的事：**理解自然语言描述，提取语义特征**。

### 输入

对模板中的**每一个属性**单独调用。输入中不包含 plan schema、topology 规则、output format。

```json
{
  "name": "background_clutter",
  "type": "boolean",
  "description": "背景是否存在大量杂波干扰（如地面纹理复杂/多物体堆叠/光影斑驳/背景有类似纸箱形状的物体）\ntrue: 背景杂，模型可能分不清前景和背景\nfalse: 背景干净，目标突出",
  "scope": "quality"
}
```

### 输出

只输出**语义特征向量**，不使用任何 runtime 术语（没有 `data_flow`、`handler`、`per_candidate`）。

```json
{
  "needs_global_context": true,
  "requires_reasoning": true,
  "candidate_level": false,
  "supports_numeric_analysis": false,
  "requires_spatial_relation": false,
  "requires_temporal_context": false,
  "semantic_type": "scene_quality",
  "reason": "背景评估必须看全图，且'杂波干扰'是语义判断不是数值分析"
}
```

### 特征向量定义（主驱动）

这些字段是 **Stage 2 Mapping 的主驱动**。不是 `semantic_type`。

| 特征 | 类型 | 说明 |
|------|------|------|
| `needs_global_context` | bool | 是否需要全图信息来判断 |
| `requires_reasoning` | bool | 是否需要语义理解而非数值计算 |
| `candidate_level` | bool | false = 场景级，结果对所有候选框生效 |
| `supports_numeric_analysis` | bool | 是否可以用数值方法（如 Laplacian/直方图）分析 |
| `requires_spatial_relation` | bool | 是否需要多个物体之间的空间关系 |
| `requires_temporal_context` | bool | 是否需要多帧时序上下文 |

**设计原则：** 当有新需求时（如多目标关系、OCR、时序动作等），你**添加新的 feature vector 字段**，而不是新增 semantic_type。Runtime 决策始终由 feature vector 驱动，`semantic_type` 不作为条件判断的依据。

### semantic_type（辅助角色——仅用于 Observability）

`semantic_type` 是给人看的，不是给代码吃的。

| 用途 | 说明 |
|------|------|
| 日志 | `[Planner] attribute=background_clutter classified as scene_quality` |
| 调试 | 在 trace 中展示分类结果，便于人工审查 |
| 分析 | 统计有多少属性被归为同一类型 |
| 不用于 | **Runtime 决策条件** |

```python
# ❌ 错误用法
if semantic_type == "scene_quality":
    data_flow = FULL

# ✅ 正确用法
if features.needs_global_context:
    data_flow = FULL
```

**为什么：** `semantic_type` 是一个 summary label，是 feature vector 的投影。它会爆炸（40+ 种），会重新耦合。而 feature vector 是正交的、可组合的。新的需求只需要加新字段，不影响既有判定逻辑。

### 备注

- Semantic Classifier 可以选择用 **Gemini**（复杂语义），也可以退化为 **关键词规则**（上线初期稳定）。
- 退化规则见附录 A。
- 每个属性独立调用，无状态依赖。

---

## Stage 2a：Capability Mapping（代码）

### 职责

将 Stage 1 输出的语义特征**确定性地**映射为 runtime 参数。这段代码没有不确定性。

### 映射规则

```python
# ─── data_flow 映射（仅使用 feature vector，不使用 semantic_type） ───

if not features.candidate_level:
    # 场景级属性一定需要全图
    data_flow = FULL
elif features.needs_global_context:
    data_flow = FULL
elif features.requires_spatial_relation:
    data_flow = FULL    # 多物体空间关系需要全图
else:
    data_flow = CROP

# ─── per_candidate 映射 ───

if features.candidate_level is False:
    per_candidate = False
else:
    per_candidate = True

# ─── required_capabilities 映射 ───
# 这里只描述"属性需要什么能力"，不决定具体用哪个 handler/model

caps = []

if features.requires_reasoning:
    caps.append("vision_reasoning")
elif features.supports_numeric_analysis:
    caps.append("numeric_analysis")
else:
    caps.append("vision_reasoning")  # 默认为视觉推理

if original_scope == "negative":
    caps.append("negative_classification")

# 未来扩展：requires_temporal_context → caps.append("temporal_analysis")
# 未来扩展：requires_spatial_relation → caps.append("spatial_relation")
```

**关键约束：** 以上所有分支中不使用 `semantic_type`。新需求只需要加 feature vector 字段并在对应位置追加 `caps.append(...)`。

### 输出

```json
{
  "attribute_key": "background_clutter",
  "data_flow": "full_image",
  "required_capabilities": ["vision_reasoning"],
  "per_candidate": false
}
```

---

## Stage 2b：Resolver（代码，全部硬编码）

### 职责

将 `required_capabilities` 解析为具体的 handler + model_id。这段代码是硬编码的映射表，没有不确定性。

### 设计原则

`required_capabilities` 描述"需要什么能力"，Resolver 决定"谁提供这个能力"。

| 能力 | 当前提供者 | 未来可替换为 |
|------|-----------|-------------|
| `vision_reasoning` | `handler=gemini, model=gemini-2.0-flash` | Claude, GPT-4o |
| `numeric_analysis` | `handler=opencv_quality` | 自定义数值模型 |
| `negative_classification` | `handler=gemini_negative, model=gemini-2.0-flash` | 专用分类器 |

### Resolver 逻辑

```python
def resolve(caps: list[str], scope: str, attribute: str) -> tuple[str, str]:
    """
    硬编码映射：capabilities → handler + model_id
    输入：capability 列表（语义层面）
    输出：handler 名称 + model_id（实现层面）
    """

    # ─── handler 映射 ───

    if "numeric_analysis" in caps:
        handler = "opencv_quality"
    elif "vision_reasoning" in caps and "negative_classification" in caps:
        handler = "gemini_negative"
    elif "vision_reasoning" in caps:
        handler = "gemini"
    else:
        handler = "gemini"  # 默认 fallback

    # ─── model_id 映射 ───

    if handler == "opencv_quality":
        model_id = "rule-engine"  # 不需要 LLM
    elif _is_high_stakes(attribute) and scope in ("semantic", "quality"):
        model_id = "gemini-2.5-pro"
    else:
        model_id = "gemini-2.0-flash"

    return handler, model_id
```

### Resolver 输出（合并到 Stage 2 最终输出）

```json
{
  "attribute_key": "background_clutter",
  "data_flow": "full_image",
  "handler": "gemini",
  "per_candidate": false,
  "model_id": "gemini-2.0-flash",
  "required_capabilities": ["vision_reasoning"],
  "prompt_key": "verify_background_clutter"
}
```

### 测试覆盖

```python
def test_scene_quality_maps_to_full_and_gemini():
    features = SemanticFeatures(semantic_type="scene_quality", ...)
    caps = capability_mapping.map(features, scope="quality")
    assert caps.data_flow == FULL
    assert caps.per_candidate is False
    assert "vision_reasoning" in caps.required_capabilities

    handler, model_id = resolver.resolve(caps, scope="quality", attribute="background_clutter")
    assert handler == "gemini"
    assert model_id == "gemini-2.0-flash"
```

---

## Stage 3：StepGraph Builder（代码）

### 职责

接收 Stage 2 输出的属性级参数，构建完整的 PipelinePlan。

### Runtime Invariant（固定拓扑，硬编码）

以下是**永远不改变**的执行顺序，不是推理问题：

```
[场景级 negative]  →  如果模板有 pure_negative 类型属性
[detect]           →  YOLO 检测
[nms]              →  非极大值抑制
[verify]           →  Gemini 验证
[质量步骤]         →  quality scope 的属性（场景级在逐候选前）
[语义步骤]         →  semantic scope 的属性
[阴性步骤]         →  negative scope 的属性（逐候选）
[merge]            →  最终合并
```

### 步骤合并规则（硬编码）

```
groupby(scope, data_flow, handler) → 同组属性合并为一个 step
```

| scope | data_flow | handler | 合并在一个 step？ |
|-------|-----------|---------|------------------|
| quality | FULL | gemini | background_clutter 独自 |
| semantic | CROP | gemini | object_type + is_package 合并 |
| negative | CROP | gemini_negative | ambiguous + open_set_negative 合并 |

### StepGraph Builder 流程图

```
Stage 2 输出:
  [pure_negative]      → FULL, 场景级, gemini_negative
  [background_clutter] → FULL, 场景级, gemini
  [object_type]        → CROP, 逐候选, gemini
  [is_package]         → CROP, 逐候选, gemini
  [ambiguous]          → CROP, 逐候选, gemini_negative
  [open_set_negative]  → CROP, 逐候选, gemini_negative

StepGraphBuilder:

  1. 拓扑骨架（固定）:
     scene_neg → detect → nms → verify → quality → semantic → negative → merge

  2. 填充属性步骤（合并分组）:
     scene_neg: [pure_negative]
     quality:   [background_clutter] (场景级, FULL, gemini)
     semantic:  [object_type, is_package] (合并, CROP, gemini)
     negative:  [ambiguous, open_set_negative] (合并, CROP, gemini_negative)

  3. 设置早退条件:
     scene_pure_negative = true → 跳过所有后续步骤

  4. 设置跳过条件:
     all(not c.exists for c in candidates) → 跳过属性/阴性步骤

  5. 输出 PipelinePlan
```

---

## Stage 4：Validator（代码）

### 职责

校验 PipelinePlan 是否合法。与 Stage 3 分离，职责分开。

### 校验项

```
必填检查（FAIL）:
  ✓ plan_id 存在
  ✓ object_name 非空
  ✓ steps 非空
  ✓ 有 detect 步骤
  ✓ 有 merge 步骤
  ✓ 所有 enabled 属性被分配到 steps

兼容检查（FAIL）:
  ✓ step_type + model_id 在 ModelRegistry 中注册
  ✓ model 的 capabilities 包含该 step_type
  ✓ 相邻 step 的 data_flow 与模型能力匹配

结构检查（WARN）:
  ○ quality 步骤在 semantic 之前
  ○ merge 在最后
  ○ nms 在 detect 之后、verify 之前
  ○ scene_negative 在 detect 之前

条件检查（WARN）:
  ○ early_exit 条件引用的变量在 executor context 中存在
  ○ skip_condition 引用的 step 存在于 plan 中
```

---

## 完整数据流示例

以 fixture 模板为例，走完整流程：

```
Fixture Template
  → TemplateParser.parse()
    → ParsedTaskSpec
      │
      ▼
Stage 1: Semantic Classifier (逐属性 LLM 调用)
  pure_negative:
    → {needs_global_context: true, requires_reasoning: false,
       candidate_level: false, semantic_type: "scene_pure_negative"}
  background_clutter:
    → {needs_global_context: true, requires_reasoning: true,
       candidate_level: false, semantic_type: "scene_quality"}
  object_type:
    → {needs_global_context: false, requires_reasoning: true,
       candidate_level: true, semantic_type: "crop_classification"}
  is_package:
    → {needs_global_context: false, requires_reasoning: true,
       candidate_level: true, semantic_type: "crop_classification"}
  ambiguous:
    → {needs_global_context: false, requires_reasoning: true,
       candidate_level: true, semantic_type: "crop_negative"}
  open_set_negative:
    → {needs_global_context: false, requires_reasoning: true,
       candidate_level: true, semantic_type: "crop_negative"}
      │
      ▼
Stage 2a: Capability Mapping (代码)
  pure_negative        → FULL,  [vision_reasoning, negative_classification], per_candidate=false
  background_clutter   → FULL,  [vision_reasoning],                          per_candidate=false
  object_type          → CROP,  [vision_reasoning],                          per_candidate=true
  is_package           → CROP,  [vision_reasoning],                          per_candidate=true
  ambiguous            → CROP,  [vision_reasoning, negative_classification], per_candidate=true
  open_set_negative    → CROP,  [vision_reasoning, negative_classification], per_candidate=true
      │
      ▼
Stage 2b: Resolver (代码, 全部硬编码)
  pure_negative        → gemini_negative,   gemini-2.0-flash
  background_clutter   → gemini,            gemini-2.0-flash
  object_type          → gemini,            gemini-2.0-flash
  is_package           → gemini,            gemini-2.0-flash
  ambiguous            → gemini_negative,   gemini-2.0-flash
  open_set_negative    → gemini_negative,   gemini-2.0-flash
      │
      ▼
Stage 3: StepGraph Builder (代码)
  → PipelinePlan（8 steps, 1 early_exit, 2 skip_conditions）
      │
      ▼
Stage 4: Validator (代码)
  → passed: true
      │
      ▼
  StepExecutor 执行
```

---

## 附录 A：Semantic Classifier 退化规则

当 LLM 不可用或降级时，Semantic Classifier 可用关键词规则替代：

```
# Feature vector 字段（主驱动）
needs_global_context = true if:
  description 含 "背景"/"环境"/"场景"/"全局"/"context"/"scene"/"全景"

requires_reasoning = true if:
  type == "boolean" 且 description 含复杂判断
  type == "multi_select" 且 option_count > 3

candidate_level = false if:
  name 含 "pure"/"scene"/"background"/"环境"
  description 含 "场景中"/"场景"

supports_numeric_analysis = true if:
  name in ("blur", "occlusion", "lighting", "brightness")

# semantic_type（仅用于 observability，不影响 runtime）
semantic_type = derive_from_feature_vector(features)
```

---

## 附录 B：与旧架构的对比

| 维度 | 旧架构 (v1) | 新架构 (v2) |
|------|------------|------------|
| LLM 输出 | 完整 PipelinePlan | 语义特征（每属性 5 个字段） |
| Prompt 长度 | ~2000 tokens（含 schema/topology/示例） | ~300 tokens（只有属性本身） |
| Runtime 逻辑 | LLM 隐含在 prompt 中 | 代码显式实现 |
| 可测试性 | LLM 输出难断言 | 每层独立单元测试 |
| 稳定性 | 高 prompt 敏感 | 低，LLM 只做语义分类 |
| 可追溯性 | LLM 给出 plan，不知道为什么 | 每步 decision 可追溯 |
| 降级策略 | 全部回退静态 plan | 可逐属性降级（LLM→关键词规则） |

---

## 附录 C：变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.0 | 2026-05-25 | 初始分层架构：4-stage Compiler（Semantic Classifier → Rule Engine → StepGraph Builder → Validator），feature vector 主驱动 |
| v2.1 | 2026-05-25 | Stage 2 拆分为 Capability Mapping + Resolver；新增 `required_capabilities` 中间层，handler/model 分离 |
