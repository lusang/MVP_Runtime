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

## Phase 2 — 架构准备

**目标：** 为后续精细化和动态 Planner 改造打好基础。

### 2.1 TemplateAttributeSpec 增加 `analysis_scope` 字段

**当前问题：** 所有 semantic 属性共享 `data_flow: CROP`，无法区分需要全图分析的属性。

**改动位置：** `schemas/template_spec.py`

```python
class TemplateAttributeSpec(BaseModel):
    # ... 现有字段 ...
    analysis_scope: Literal["crop", "full_image"] = Field(
        default="crop",
        description="crop=在检测框局部判断, full_image=需要全图上下文",
    )
```

**依赖关系：** 需要在 DataHub 侧模板和 MVP 的 `TemplateParser` 中都支持此字段。

### 2.2 提示词从常量抽成可配置模板

**当前问题：** `models/gemini_client.py` 中的 `_VERIFY_*_PROMPT` 和 `_MERGE_PROMPT` 是硬编码模块常量，修改需要改代码。

**方案：** 将提示词模板移到独立的 `prompts/` 目录，每个提示词一个文件，支持通过环境变量或请求参数覆盖。

**改动的文件：**
- 新增 `prompts/verify_object.txt`
- 新增 `prompts/verify_attribute.txt`
- 新增 `prompts/verify_scene_negative.txt`
- 新增 `prompts/merge.txt`
- 新增 `prompts/planner.txt`
- 修改 `models/gemini_client.py` — 加载文件而非常量
- 可选：`prompts/` 目录增加 JSON 元数据文件（prompt 版本、描述等）

**加载逻辑：**
```python
class PromptManager:
    def load(self, name: str) -> str:
        # 1. 优先使用环境变量 MVP_PROMPT_{NAME}
        # 2. 回退到 prompts/{name}.txt 文件
        # 3. 支持 {placeholder} 格式化
```

### 2.3 Planner 按 `analysis_scope` 分组属性到不同 step

**当前静态计划**（`runtime/planner.py:_StaticPlanFactory`）将所有属性集中在一个 attribute step。

**改造后：**
- `analysis_scope="crop"` 的属性 → attribute step with `data_flow: CROP`
- `analysis_scope="full_image"` 的属性 → attribute step with `data_flow: FULL`
- 可选的多个 attribute step 按 order 排列

**同时改造 `StepExecutor`：**
- `_run_attribute` 根据 step 的 `data_flow` 选择使用 crop 图还是全图
- 增加 `_run_full_image_attribute` 分支

---

## Phase 3 — 真实调用

**目标：** 连接真实模型，获取有意义的 trace 数据用于调优。

### 3.1 启用真实 Gemini 调用

- [ ] 配置真实 `GEMINI_API_KEY`
- [ ] 设置 `MVP_FORCE_GEMINI_MOCK=0`
- [ ] 运行 pipeline，确认真实 Gemini 调用成功
- [ ] 添加 API 错误重试 / fallback 逻辑（当前 mock 模式没有覆盖）
- [ ] 确认 token 消耗可控

### 3.2 接入 LangFuse

**为什么放在这里：** LangFuse 记录 prompt/response/cost 才有调优价值。在 mock 模式下接入是无意义的。

- [ ] 集成 `langfuse` Python SDK
- [ ] 用 `@observe()` 装饰器包装 `GeminiClient` 中的每次 API 调用
- [ ] 关联 trace_id = run_id，将一次 pipeline 执行的所有 LLM 调用串联成一个 trace
- [ ] 在 LangFuse UI 上确认 trace 结构完整
- [ ] 配置 token 用量统计

### 3.3 根据 Trace 调优提示词

- [ ] 收集至少 20 张测试图的执行结果
- [ ] 分析 LangFuse trace 中每步 Gemini 的输出质量
- [ ] 修改 `prompts/*.txt` 中的提示词，重新运行对比
- [ ] 建立简单的对比矩阵（每次改动前后指标）

---

## Phase 4 — Merge 增强

**目标：** 当一张图出现多个候选框时，merge 层能做出更合理的决策。

### 4.1 NMS 去重（确定性规则）

- [ ] 在 merge 前插入 NMS（Non-Maximum Suppression）步骤
- [ ] 配置 IoU 阈值（如 0.5），高于阈值的重叠框只保留置信度最高的一个
- [ ] 被抑制的候选框标记为 `rejected` 而非删除（保留审计轨迹）

### 4.2 加权投票规则

- [ ] 配置 detector 和 verifier 的权重（如 detect:verify = 3:7）
- [ ] 当两者冲突时（detect 高分 but verify rejected），按权重决策
- [ ] 合并表决策略可配置（YAML 或 JSON 配置文件）

### 4.3 属性冲突解决

- [ ] 同一属性多个候选框都给出了有效值 → 取置信度最高的
- [ ] 属性值置信度过低（< threshold）→ 标记为 `uncertain` 而非丢弃

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
| 1.2 | 验证 handler 输出 | 低 | 无 | 高 | P0 |
| 2.1 | analysis_scope 字段 | 低 | DataHub 侧配合 | 高 | P0 |
| 2.2 | 提示词配置化 | 中 | 无 | 中 | P1 |
| 2.3 | Planner 按 scope 分组 | 中 | 2.1 | 高 | P0 |
| 3.1 | 启用真实 Gemini | 低 | 无 | 高 | P1 |
| 3.2 | 接入 LangFuse | 中 | 3.1 | 中 | P1 |
| 3.3 | 调优提示词 | 高 | 3.1, 2.2 | 高 | P1 |
| 4.1 | NMS 去重 | 中 | 无 | 中 | P2 |
| 4.2 | 加权投票 | 中 | 无 | 中 | P2 |
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
