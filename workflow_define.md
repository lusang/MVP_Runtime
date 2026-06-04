# MVP Runtime — Workflow Architecture (Target Design)

> 版本：v1.1  
> 最后更新：2026-05-25  
> 状态：设计目标，待实现

---

## 核心思想

当前 workflow 是"StepExecutor 顺序执行步骤，状态隐式，错误被吞"。
目标 workflow 是 **"DAG Runtime with Explicit State Machine + Structured Execution Context + Full Observability"**。

```
当前 (现状):
  plan → for step in steps:
    dispatch(step)
    → 偷偷改 c.exists / 偷偷 skip / 异常被吞
    → 不知道最终状态是什么

目标 (设计):
  plan → DAG (可视) → for step in steps:
     ├─ 1. 读取 RuntimeState (只通过 state 读写 candidates)
     ├─ 2. 创建 StepContext
     ├─ 3. dispatch(step) → 产出 StepResult (status, reason, events)
     ├─ 4. StepResult 驱动下一步 (skip / continue / early_exit 全部显式)
     ├─ 5. 写入 DB
     └─ 6. RuntimeState 进入下一个不可变版本
```

---

## 1. Candidate State Model（最高优先级）

### 为什么必须做

当前 `c.exists` 是隐式状态，多个 step 以不同方式修改它：

```
verify reject  → c.exists = False       # 但不知道是被 verify 拒绝的
nms suppress   → c.exists = False       # 还是被 nms 抑制的
skip           → 根本不进循环            # 还是被跳过的
```

后面会出现：*"为什么 merge 收到了这个 candidate？"* — 根本查不出来。

### 设计

```python
from enum import Enum

class CandidateState(str, Enum):
    """Candidate 的显式状态机。Runtime 唯一真相源。"""
    DETECTED   = "detected"    # YOLO 检测到，等待 NMS
    SUPPRESSED = "suppressed"  # NMS 抑制，不参与后续
    VERIFIED   = "verified"    # 验证通过，可以走 quality/semantic
    REJECTED   = "rejected"    # 验证拒绝，只走 negative
    NEGATIVE   = "negative"    # 阴性判定触发
    MERGED     = "merged"      # 已合并，最终状态
```

### 状态转换图

```
DETECTED
  │
  ├── NMS 抑制 → SUPPRESSED (跳过后续所有)
  │
  └── NMS 保留
       │
       ├── verify ok= true  → VERIFIED
       │    ├── quality/semantic 正常
       │    └── negative 检查
       │         └── 触发 → NEGATIVE
       │
       └── verify ok= false → REJECTED
            └── 只走 negative 检查
                 └── 触发 → NEGATIVE

最终:
  VERIFIED / NEGATIVE / REJECTED → MERGED
```

### 约束

```python
# 合法的状态转换（由每个 step 的 _run_* 方法执行）
ALLOWED_TRANSITIONS = {
    CandidateState.DETECTED:   {CandidateState.SUPPRESSED, CandidateState.VERIFIED, CandidateState.REJECTED},
    CandidateState.SUPPRESSED: set(),                          # 终态
    CandidateState.VERIFIED:   {CandidateState.NEGATIVE, CandidateState.MERGED},
    CandidateState.REJECTED:   {CandidateState.NEGATIVE, CandidateState.MERGED},
    CandidateState.NEGATIVE:   {CandidateState.MERGED},
    CandidateState.MERGED:     set(),                          # 终态
}
```

### Candidate（更新后）

```python
@dataclass
class Candidate:
    object_id: str
    state: CandidateState = CandidateState.DETECTED

    # 分数
    detector_score: float = 0.0
    verify_score: float = 0.0
    confidence: float = 0.0

    # 空间
    bbox: BBox
    analysis_path: str
    analysis_bbox: BBox

    # 各 step 产出（按 scope 分拆）
    quality: dict = field(default_factory=dict)         # quality step 产出
    attributes: dict = field(default_factory=dict)       # semantic step 产出
    negative_flags: dict = field(default_factory=dict)   # negative step 产出

    # 可行性（quality 判定）
    missing_attributes: list = field(default_factory=list)

    # 变更历史（每次 state 变更追加一条）
    history: list = field(default_factory=list)

    def transition_to(self, new_state: CandidateState, step: str, reason: str):
        assert new_state in ALLOWED_TRANSITIONS[self.state], \
            f"Illegal transition: {self.state} → {new_state}"
        old_state = self.state
        self.history.append({
            "step": step,
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        self.state = new_state
```

### 当前代码的对应修改

```
当前                             目标
─────────────────────────────────────────────────────
c.exists = False                 c.transition_to(CandidateState.SUPPRESSED, "nms", ...)
c.exists = False                 c.transition_to(CandidateState.REJECTED, "verify", ...)
c.exists = True                  c.transition_to(CandidateState.VERIFIED, "verify", ...)
if not c.exists: continue        if c.state in (SUPPRESSED,): continue
```

---

## 2. Execution Decision Standardization

### 问题

当前 skip / early exit 是隐式的：

```python
# step_executor.py 中散落着
if not c.exists:
    continue
if ctx.scene_pure_negative:
    break
```

这些 `continue` / `break` 没有产生任何记录。后面 DAG 可视化做出来，也会和实际执行不一致。

### 设计

每个 step 执行后产出 `StepResult`，所有执行决策都显式事件化。

```python
@dataclass
class StepResult:
    """每个 step 执行后的结构化结果。"""
    step: str                  # step_type
    status: str                # "success" | "skipped" | "failed" | "early_exit"
    reason: str                # 为什么 skip / 为什么失败
    latency_ms: float
    input_count: int           # 进入该 step 时的 candidate 数
    output_count: int          # 离开该 step 时的 candidate 数
    error: str | None = None
    events: list[dict] = field(default_factory=list)  # 该 step 产生的 runtime_step_event
```

### 所有 skip / early exit 必须显式

```python
# execute() 主循环
for step in sorted(plan.steps, key=lambda s: s.order):
    result = await self._execute_one(step, ctx, runtime_state, ...)
    results.append(result)

    # 所有决策都从 StepResult 读取，而不是散落在各处
    if result.status == "early_exit":
        emit_event("pipeline.early_exit", {"source": step.step, "reason": result.reason})
        break
    elif result.status == "skipped":
        emit_event("pipeline.skip", {"step": step.step, "reason": result.reason})
        continue
```

```python
async def _execute_one(self, step, ctx, runtime_state, ...) -> StepResult:
    # 1. 检查跳过条件（返回显式的 skipped result）
    skip_reason = self._check_skip_conditions(step, runtime_state)
    if skip_reason:
        return StepResult(
            step=step.step, status="skipped",
            reason=skip_reason, ...
        )

    # 2. 检查早退条件
    exit_reason = self._check_early_exit(step, runtime_state)
    if exit_reason:
        return StepResult(
            step=step.step, status="early_exit",
            reason=exit_reason, ...
        )

    # 3. 正常执行（可能失败）
    try:
        await self._dispatch(step, ctx, runtime_state, ...)
        return StepResult(step=step.step, status="success", ...)
    except Exception as e:
        return StepResult(
            step=step.step, status="failed",
            reason=str(e), ...
        )
```

### 对比

```
当前                             目标
─────────────────────────────────────────────
if condition: continue           skip 显式 → StepResult(status="skipped", reason=...)
if condition: break              early_exit 显式 → StepResult(status="early_exit", ...)
except Exception: pass           失败显式 → StepResult(status="failed", error=...)
```

---

## 3. RuntimeState — 显式共享状态

### 问题

当前 `candidates` 是 `list[CandidateState]`，所有 step 直接 `ctx.candidates.append()` / `ctx.candidates[*].exists = False`。

这导致：
- 不知道谁改了 candidate
- replay 很难（要回溯所有 mutation）
- 多个 step 之间存在隐式耦合

### 设计

```python
@dataclass
class RuntimeState:
    """Step 之间共享的 Runtime 状态。所有 step 通过 state 读写 candidates。"""
    candidates: list[Candidate]
    scene_flags: dict[str, Any]          # scene_pure_negative, ...
    metrics: dict[str, Any]              # 跨 step 的数值指标
    artifacts: dict[str, Any]            # 中间产物（如 crop 路径、debug 图片）

    def candidate_by_id(self, oid: str) -> Candidate | None:
        return next((c for c in self.candidates if c.object_id == oid), None)

    def active_candidates(self) -> list[Candidate]:
        """未被抑制的候选框（当前 step 应该处理的）。"""
        return [c for c in self.candidates
                if c.state not in (CandidateState.SUPPRESSED,)]
```

### Step 签名变更

```python
# 当前
async def _run_verify(self, step, ctx, image_path, parsed):
    for c in ctx.candidates:          # 直接 mutable 访问
        c.exists = False               # 隐式状态变更

# 目标
async def _run_verify(self, step, ctx, runtime_state: RuntimeState, ...):
    for c in runtime_state.active_candidates():  # 通过 state 读取
        c.transition_to(CandidateState.REJECTED, "verify", ...)  # 显式状态变更
        # 或
        c.transition_to(CandidateState.VERIFIED, "verify", ...)
```

### 原则

- **读**：通过 `runtime_state.active_candidates()` / `runtime_state.candidate_by_id()`
- **写**：通过 `c.transition_to()` 修改 state，通过 `runtime_state.metrics` / `runtime_state.scene_flags` 写共享数据
- **不变性**：不允许 `some_list.append(c)` 或 `c.attr = x` 直接修改

---

## 4. StepContext — 结构化执行上下文

### 为什么需要

当前 step 失败时你不知道：

```
✅ 你真正需要知道的     ❌ 现在可能缺失
是哪个 DAG node        ❌
属于哪个 attribute     ❌
输入是什么             ❌
前置 step 是谁         ❌
使用了哪个 model       ❌
prompt 是哪个版本      ❌
candidate 数量         ❌
crop 来源              ❌
重试过几次             ❌
```

### 设计

```python
@dataclass
class StepContext:
    """每个 step 执行前创建，贯穿整个 step 生命周期。"""
    run_id: str
    step_id: str
    step_type: str
    step_order: int
    attribute_keys: list[str] | None
    handler: str
    model_id: str
    prompt_version: str
    upstream_steps: list[str]
    input_candidate_count: int
    started_at: float
    finished_at: float | None = None
    error_message: str | None = None
    retry_count: int = 0
```

### 使用方式

```python
ctx = StepContext(
    run_id=run_id,
    step_id=f"{step.step}_step_{step.order}",
    step_type=step.step,
    step_order=step.order,
    attribute_keys=step.params.get("attribute_keys"),
    handler=resolve_handler(step),
    model_id=step.model_id,
    prompt_version=resolve_prompt_version(step),
    upstream_steps=resolve_upstream_steps(plan, step),
    input_candidate_count=len(runtime_state.active_candidates()),
    started_at=time.perf_counter(),
)

result = await self._execute_one(step, ctx, runtime_state, ...)
ctx.finished_at = time.perf_counter()
```

---

## 5. DAG 可视化 + Versioning

### 目标

**每次 Runtime 执行，自动生成 DAG 拓扑并持久化。**

### DAG 结构

当前拓扑是固定的（非 LLM 生成）：

```
scene_negative ── early_exit ──→ (跳过全部)
    │
    ▼
detect ──→ crop ──→ NMS
                       │
                       ▼ (suppressed)
                    verify ──→ rejected ──→ negative
                       │
                       ▼ (verified)
                    quality(background_clutter)
                       │
                       ▼
                    semantic(object_type, is_package)
                       │
                       ▼
                    negative(ambiguous, open_set_negative)
                       │
                       ▼
                    merge
```

### graph_hash

```python
graph_hash = hashlib.sha256(
    json.dumps(dag_snapshot, sort_keys=True).encode()
).hexdigest()[:16]

# 用途:
# 1. runtime_run.graph_hash — 记录本次执行使用的 DAG 版本
# 2. runtime_plan_snapshot.graph_hash — 同 graph_hash 可复用已有 snapshot
# 3. 后续可以 diff: graph_hash_v1 vs graph_hash_v2 → 知道拓扑变了什么
```

### DAG Snapshot 内容

```python
dag_snapshot = {
    "graph_hash": "...",
    "plan_hash": "...",
    "nodes": [...],    # 每个 step 的定义
    "edges": [...],    # 数据流向
    "skip_rules": [...],    # 显式的 skip 规则（graph 的一部分）
    "early_exit_rules": [...],  # 显式的 early_exit 规则
}
```

**关键**：skip/early_exit 规则是 graph 的一部分，不只是 executor 中的 `if` 语句。

### 可视化输出（文本格式）

```
[0] scene_negative  [gemini-2.0-flash]
     │  early_exit → scene_pure_negative = true
     ▼
[1] detect  [yolo-world-v2-x]
     ▼
[2] nms  [rule-engine]
     │  skip_if → all candidates suppressed
     ▼
[3] verify  [gemini-2.0-flash]
     │  rejected → [negative]
     │  verified → [quality]
     ▼
[4] quality  [gemini-2.0-flash] (background_clutter)
     ▼
[5] semantic  [gemini-2.5-pro] (object_type, is_package)
     ▼
[6] negative  [gemini-2.0-flash] (ambiguous, open_set_negative)
     ▼
[7] merge  [gemini-2.0-flash]
```

---

## 6. Unified Step Trace

### 目标

**所有 step（不只有 Gemini）统一 trace，统一存储。**

### Span 结构

```
run (trace)
  ├── planner     → latency, step_count, graph_hash
  ├── detect      → candidate_count, latency
  ├── nms         → before/after count, iou_threshold
  ├── verify      → ok/fail ratio, latency
  ├── quality     → per-attribute scores
  ├── semantic    → per-attribute results, confidence
  ├── negative    → per-flag results
  └── merge       → positive/negative ratio
```

### 与 Langfuse 的关系

```
Unified Step Trace 不替代 Langfuse，而是互补：

  Langfuse → LLM 级别的详细追踪（token 数、prompt 文本、response）
  runtime_step_event → DAG 级别的结构化追踪（所有 step 的全貌）

两者通过 run_id 关联。
```

---

## 7. Error Handling Policy

### 当前

```
所有异常 → except Exception: pass → success=False
结果: 不知道哪个 step 失败、为什么失败
```

### 目标

```python
class StepErrorPolicy:
    IGNORE = "ignore"              # 记录错误，继续 pipeline
    SKIP_DOWNSTREAM = "skip"       # 记录错误，跳过下游 step
    FAIL_PIPELINE = "fail"         # 记录错误，终止整个 pipeline

STEP_ERROR_POLICIES = {
    "detect":   StepErrorPolicy.FAIL_PIPELINE,
    "nms":      StepErrorPolicy.IGNORE,
    "verify":   StepErrorPolicy.IGNORE,
    "quality":  StepErrorPolicy.IGNORE,
    "semantic": StepErrorPolicy.IGNORE,     # 注意：不是 SKIP_DOWNSTREAM
    "negative": StepErrorPolicy.IGNORE,
    "merge":    StepErrorPolicy.FAIL_PIPELINE,
}
```

**为什么 semantic 不是 SKIP_DOWNSTREAM？** 因为 semantic 失败不代表不能 merge —— 可能只是 object_type 失败，但 blur、negative、verify 都成功。后面可以 partial merge。当前不要写死，先保留为 IGNORE，等 attribute 级错误处理再做细化。

---

## 8. 完整数据流（最终目标）

```
AsyncWorker.process_batch()
  │
  ├─ TemplateParser.parse(template)         → ParsedTaskSpec
  ├─ Planner.compile(parsed)                → PipelinePlan
  ├─ DAG.visualize(plan)                    → dag_snapshot + graph_hash
  │
  └─ for each task:
       ├─ resolve_url(task.url)             → image_path
       │
       └─ engine.run_with_plan(image, plan, parsed)
            │
            ├─ 1. 创建 RuntimeRun (写入 runtime_run)
            │     run_id, graph_hash, plan_hash, status="running"
            │
            ├─ 2. 保存 DAG Snapshot (写入 runtime_plan_snapshot)
            │     graph_hash, graph_json
            │
            ├─ 3. 初始化 RuntimeState
            │     candidates=[], scene_flags={}, metrics={}
            │
            └─ 4. StepExecutor.execute()
                  │
                  ├─ for step in plan.steps:
                  │     ├─ 创建 StepContext
                  │     ├─ 写入 runtime_step (status="running")
                  │     ├─ _execute_one() → StepResult
                  │     │   ├─ skip 检查 → StepResult(status="skipped", reason)
                  │     │   ├─ early_exit → StepResult(status="early_exit", reason)
                  │     │   ├─ dispatch → success → StepResult(status="success")
                  │     │   └─ dispatch → fail   → StepResult(status="failed", error)
                  │     ├─ 更新 runtime_step
                  │     └─ 写入 runtime_step_event (每个事件)
                  │
                  └─ 更新 runtime_run (status, total_latency)
```

---

## 9. 与现有架构的关系

```
当前文件                     目标变化
─────────────────────────────────────────────────
schemas/candidate_state.py   新增 CandidateState 枚举 + Candidate dataclass + transition_to()
runtime/step_executor.py     新增 RuntimeState, StepContext, StepResult, 显式 skip/early_exit
runtime/engine.py            新增 DAG 可视化、RuntimeState 初始化、DB 写入
runtime/planner.py           新增 graph_hash 生成
runtime/performance_tracker.py  保留或由 event 替代
runtime/tracer.py            保留 Langfuse LLM trace，新增 DAG 级 span
models/gemini_client.py      Prompt 去 Runtime 化（merge prompt 精简）
models/gemini_merger.py      读取 Candidate.state 而非隐式 exists
runtime/nms.py               c.transition_to(SUPPRESSED) 而非 c.exists = False
                             新文件: dag_visualizer.py
                             新文件: db_recorder.py (异步写入)
config/mvp_runtime_events_ddl.sql  已就绪
```

---

## 10. 优先级与实施路径

| Priority | 项目 | 前置 | 为什么是这个顺序 |
|----------|------|------|----------------|
| **P1** | Candidate State Model | 无 | 没有它，所有 observability 都没用——数据流本身是隐式的 |
| **P2** | RuntimeState + StepResult | P1 | state 到位后，才能统一读写路径 + 显式化 skip/early_exit |
| **P3** | DAG 可视化 + graph_hash | P2 | 有了显式 state 和 StepResult，DAG 才能反映真实执行 |
| **P4** | Unified Step Trace + DB | P3 | DAG + state + result 都有了，写入 DB 自然补上 |
| **P5** | Prompt 去 Runtime 化 | 无 | 独立于上述，可并行 |

### 现在不要做

```
❌ 真 DAG 调度器（不必要，当前线性执行够用）
❌ 分布式 Runtime（不必要，MVP 不需要）
❌ 动态 topology（不必要，当前拓扑固定）
❌ Event sourcing（太重了，RuntimeState 够用）
❌ Attribute 级错误处理（未来再做，当前不要写死）
```
