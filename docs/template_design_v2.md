# Template Design v2 — 分层标注框架

> 基于 eval 数据反推，40% 匹配率的根因不是工程实现，而是模板设计与模型能力的语义错位。
> 人认为"重要"的标签 ≠ 模型能可靠学习的标签。

---

## 一、现状分析：当前模板的四个问题

### 问题 1：主观离散标签不可学习

当前把所有属性都设计为离散枚举（single_select / multi_select）：

```json
{
  "name": "blur",
  "type": "single_select",
  "options": ["clear", "slight", "heavy"]
}
```

但"轻微模糊"和"严重模糊"之间的边界是人为定义的，**模型看到的只是连续像素**。同一个 blur 值在不同光照条件下的人类判断不一致，导致：

| 属性 | 准确率 | 问题描述 |
|------|--------|---------|
| blur | 36.8% | human split 最主观，模型最不稳定 |
| occlusion | 47.4% | "partial" vs "heavy" 边界模糊 |
| lighting | 57.9% | "dim" vs "harsh" 取决于参照系 |

**结论：** 可观测性属性应设计为连续量（0.0-1.0），由 OpenCV 出分，业务侧定义阈值。

### 问题 2：多选导致过生成

```json
{
  "name": "object_type",
  "type": "multi_select",
  "options": ["cardboard_box", "envelope", "poly_mailer", "bag", ...]
}
```

Gemini 倾向于"全部列出"而非"只选最匹配的"。即使加了 confidence 约束，50% 的情况下仍然多选。

**结论：** 多选适合**人**做标注（人知道自己在看什么），但不适合**模型**做推理（模型没有"这是唯一正确标签"的置信感）。多选应限制在特征对象的核心属性（Layer 1），且类型尽量改为 single_select。

### 问题 3：场景级与目标级未分离

- `scenario`（场景描述）、`person_action`（行为描述）是场景级的
- `size_category`、`blur`、`occlusion` 是目标级的

当前全部走 per-candidate 精度，浪费模型调用。

**结论：** 场景级属性（Layer 2）不应参与 per-candidate 流程，应在全图分析阶段完成，且可以使用更强模型而不影响主线延迟。

### 问题 4：negative 标签未区分守卫与分析

| 标签 | 定位 | 实际作用 |
|------|------|---------|
| Pure Negative | 场景级守卫 | 影响流程走向（early exit） |
| Hard Negative | 候选级守卫 | 影响目标是否被 reject |
| Ambiguous | 分析用 | 不改变流程，仅辅助判断 |
| Open-set Negative | 分析用 | 不改变流程，仅用于统计 |

四种布尔值混合在一个 scope 中，导致 Planner 无法区分"守卫"和"分析"。

**结论：** 守卫型 negative（Pure Negative, Hard Negative）应作为 Layer 1 的流程控制，分析型 negative（Ambiguous, Open-set Negative）应归入 Layer 2。

---

## 二、新架构：双层标注框架

```
                    ┌────────────────────────────────┐
                    │          Template v2            │
                    │  (分层设计，每层职责明确)          │
                    └────────────┬───────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                      ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │  Layer 1             │           │  Layer 2             │
    │  模型可学习特征       │           │  业务分析标签         │
    │  (Planner 决策依据)   │           │  (后处理 + 统计)      │
    ├─────────────────────┤           ├─────────────────────┤
    │ • 核心特征            │           │ • 场景级属性          │
    │   - is_package       │           │   - scenario         │
    │   - object_type      │           │   - person_action    │
    │   - package_form     │           │   - size_category    │
    │   - brand_logo       │           │                      │
    │ • 可观测性 (连续量)    │           │ • 分析型 negative    │
    │   - blur (0.0-1.0)   │           │   - ambiguous        │
    │   - occlusion (0.0-1.0)│         │   - open_set_negative│
    │   - lighting (0.0-1.0)│         │                      │
    │ • 守卫型 negative     │           │ • 附加统计标签        │
    │   - pure_negative     │           │   - (扩展预留)       │
    │   - hard_negative     │           │                      │
    ├─────────────────────┤           ├─────────────────────┤
    │ 影响流程             │           │ 不影响流程            │
    │ 需要高精度           │           │ 可容忍低精度          │
    │ OpenCV + Gemini 联动 │           │ Gemini 后处理即可     │
    │ per-candidate 粒度   │           │ 全图/场景级粒度       │
    └─────────────────────┘           └─────────────────────┘
```

### Layer 1 — 模型可学习特征

**设计原则：**
1. **客观可测量** — 属性值应是模型能稳定输出的，不需要主观判断
2. **类型匹配** — 二分类/单选 → Gemini；连续量 → OpenCV；布尔守卫 → 流程控制
3. **影响流程** — Layer 1 的输出决定是否继续、是否 reject、是否 fallback

**属性设计规范：**

| 子层 | 类型 | 推荐 handler | 示例 |
|------|------|-------------|------|
| 核心特征 | boolean / single_select / multi_select | gemini | is_package, object_type, brand_logo |
| 可观测性 | numeric (0.0-1.0) | opencv_quality | blur_score, occlusion_ratio, lighting_exposure |
| 守卫 | boolean | gemini_negative | pure_negative, hard_negative |

**可观测性属性的迁移方式：**

```
旧: blur = single_select {clear, slight, heavy}
新: blur_score = numeric {range: [0.0, 1.0], higher_is: "clearer"}
    → OpenCV Laplacian 方差归一化出分
    → 业务侧定义: score > 0.6 = "clear", 0.3-0.6 = "slight", < 0.3 = "heavy"

旧: occlusion = single_select {none, partial, heavy}
新: occlusion_ratio = numeric {range: [0.0, 1.0], higher_is: "more occluded"}
    → 基于边缘连续性 / 颜色直方图差异出分
    → 业务侧定义: ratio < 0.2 = "none", 0.2-0.6 = "partial", > 0.6 = "heavy"

旧: lighting = single_select {normal, dim, harsh}
新: exposure_value = numeric {range: [0.0, 1.0], higher_is: "brighter"}
    → 图像亮度直方图均值出分
    → 业务侧定义: value > 0.6 = "normal", 0.3-0.6 = "dim", < 0.3 = "harsh"
```

### Layer 2 — 业务分析标签

**设计原则：**
1. **不参与主线决策** — Layer 2 的结果不改变 pipeline 流程
2. **场景级分析** — 全图输入，不对每个 candidate 单独调用
3. **强模型后处理** — 可以使用 gemini-2.5-pro，不影响主线延迟
4. **可扩展** — 业务可以按需添加，不影响模型训练标签

**推荐执行方式：**

```
detect / verify / Layer 1 完成
    ↓
Layer 2 全图分析（可选、异步、强模型）
    ↓
业务层消费: scenario=delivery, person_action=carrying
            用于报表/搜索/过滤，不用于模型训练
```

---

## 三、Pipeline 结构变化

### 当前（单层，所有属性平铺）

```
scene_neg → full_quality → full_attribute → full_negative → merge
                          (全部 per-candidate)
```

### 新（双层，Layer 1 驱动主线 + Layer 2 可选后处理）

```
                              ┌──────────────────────┐
                              │  场景守卫检查          │
                              │  pure_negative?       │
                              │  → 是 → early exit    │
                              │  → 否 → 继续          │
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  可观测性分析          │
                              │  blur_score: 0.0-1.0  │
                              │  occlusion_ratio      │
                              │  → OpenCV, 不调LLM    │
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  核心特征推理          │
                              │  object_type          │
                              │  brand_logo           │
                              │  package_form         │
                              │  → Gemini crop-level  │
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  候选守卫检查          │
                              │  hard_negative?       │
                              │  → 是 → reject        │
                              │  → 否 → 接受          │
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  Layer 2 (可选)        │
                              │  scenario             │
                              │  person_action        │
                              │  size_category        │
                              │  → 全图 Gemini 后处理  │
                              └──────────────────────┘
```

### Planner 决策逻辑变化

| 当前 | 新 |
|------|-----|
| 按 scope 分组（quality/semantic/negative） | 按 layer + 子层分组 |
| 所有属性平铺到一个步骤 | Layer 1 严格排序、Layer 2 可选 |
| quality 用 OpenCV，semantic 用 Gemini | 可观测性→OpenCV，核心→Gemini，守卫→流程控制 |
| handler_map 只是传递 | Planner 根据属性类型决定 pipeline 拓扑 |

---

## 四、模板 Schema 变化

### 当前结构（扁平）

```json
{
  "attributes": [
    { "name": "package_form", "type": "multi_select", "scope": "semantic" },
    { "name": "size_category", "type": "single_select", "scope": "semantic" },
    { "name": "brand_logo", "type": "single_select", "scope": "semantic" }
  ],
  "quality": {
    "attributes": [
      { "name": "occlusion", "type": "single_select" },
      { "name": "blur", "type": "single_select" },
      { "name": "lighting", "type": "single_select" }
    ]
  },
  "negative": {
    "attributes": [
      { "name": "Pure Negative" }, { "name": "Hard Negative" },
      { "name": "Ambiguous" }, { "name": "Open-set Negative" }
    ]
  }
}
```

### 新结构（分层）

```json
{
  "object_name": "Package",
  "description": "...",
  "layer_1": {
    "description": "Model-learnable features — drive pipeline decisions",
    "core_features": {
      "description": "Core identity attributes of the target object",
      "attributes": [
        {
          "name": "is_package",
          "type": "boolean",
          "description": "Does this detection correspond to an actual package?",
          "handler": "gemini"
        },
        {
          "name": "object_type",
          "type": "single_select",
          "description": "Primary packaging type — select the single best match",
          "options": ["cardboard_box", "envelope", "poly_mailer", "bag", "irregular"],
          "handler": "gemini"
        },
        {
          "name": "package_form",
          "type": "single_select",
          "description": "Physical form of the package surface",
          "options": ["box", "envelope", "bag", "soft_package"],
          "handler": "gemini"
        },
        {
          "name": "brand_logo",
          "type": "single_select",
          "description": "Logistics brand logo visible on package",
          "options": ["amazon", "fedex", "ups", "dhl", "generic"],
          "handler": "gemini",
          "optional": true
        }
      ]
    },
    "observability": {
      "description": "Continuous physical quantities — OpenCV measured, thresholded by business",
      "attributes": [
        {
          "name": "blur_score",
          "type": "numeric",
          "range": [0.0, 1.0],
          "higher_is": "clearer",
          "description": "Normalized Laplacian variance of the target region",
          "handler": "opencv_quality"
        },
        {
          "name": "occlusion_ratio",
          "type": "numeric",
          "range": [0.0, 1.0],
          "higher_is": "more occluded",
          "description": "Estimated ratio of target boundary continuity loss",
          "handler": "opencv_quality"
        },
        {
          "name": "exposure_value",
          "type": "numeric",
          "range": [0.0, 1.0],
          "higher_is": "brighter",
          "description": "Normalized mean brightness of the target region",
          "handler": "opencv_quality"
        }
      ]
    },
    "guardrails": {
      "description": "Boolean guards that control pipeline flow",
      "attributes": [
        {
          "name": "pure_negative",
          "type": "boolean",
          "description": "No target object exists in the scene at all",
          "handler": "gemini_negative",
          "scope": "scene"
        },
        {
          "name": "hard_negative",
          "type": "boolean",
          "description": "Detection is visually similar to target but is not actually one",
          "handler": "gemini_negative",
          "scope": "candidate"
        }
      ]
    }
  },
  "layer_2": {
    "description": "Business analytics — scene-level post-processing, no pipeline impact",
    "attributes": [
      {
        "name": "scenario",
        "type": "single_select",
        "options": ["delivery", "resident", "commercial", "outdoor"],
        "description": "Scene context for downstream analytics",
        "scope": "scene"
      },
      {
        "name": "person_action",
        "type": "single_select",
        "options": ["carrying", "manipulating", "none"],
        "description": "Person interaction with the package",
        "scope": "scene"
      },
      {
        "name": "size_category",
        "type": "single_select",
        "options": ["small", "medium", "large"],
        "description": "Qualitative size category for business filtering",
        "scope": "scene"
      },
      {
        "name": "ambiguous",
        "type": "boolean",
        "description": "Detection is ambiguous even to human reviewers",
        "scope": "scene",
        "optional": true
      },
      {
        "name": "open_set_negative",
        "type": "boolean",
        "description": "Detection is a novel/unseen packaging form",
        "scope": "scene",
        "optional": true
      }
    ]
  }
}
```

---

## 五、迁移路径

### Phase 1 — 模板改造（无代码变更）
- 编写新模板 JSON（符合 v2 schema）
- 保持旧模板兼容（Parser 兼容双版本）
- 确认 Layer 2 属性不影响 pipeline 主线

### Phase 2 — Pipeline 适配
- Layer 1 的可观测性采用 `numeric` 类型 + OpenCV handler
- Layer 1 的守卫 `scope: "scene"` 触发场景级检查 + early exit
- Layer 2 作为可选的后处理步骤（独立 Gemini 调用）

### Phase 3 — Planner 升级
- 根据 template v2 schema 自动决策 pipeline 拓扑
- Layer 2 属性不进 per-candidate 循环
- 可观测性属性跳过 Gemini 直接使用 OpenCV 分数

---

## 六、预期收益

| 指标 | 当前 | 预期 |
|------|------|------|
| blur 准确率 | 36.8% | > 80%（OpenCV 连续量，阈值可控） |
| occlusion 准确率 | 47.4% | > 80%（同上） |
| lighting 准确率 | 57.9% | > 80%（同上） |
| object_type 准确率 | 50.0% | > 70%（single_select + 强制单选） |
| 平均延迟 | 24.6s | < 15s（Layer 2 可异步，主线更快） |
| 模板可扩展性 | 差（改属性要改代码） | 好（描述驱动 Planner 自动适配） |
