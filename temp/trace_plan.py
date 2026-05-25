"""
Trace Planner.compile() output for the real fixture template.
"""
import json, sys
sys.path.insert(0, '.')

from runtime.planner import compile_plan
from runtime.template_parser import TemplateParser

fixture = json.loads(open("test/fixtures/request_151049_1tasks.json", encoding="utf-8").read())
parsed = TemplateParser().parse(fixture["template"])
plan = compile_plan(parsed)

print("=== Template Info ===")
print(f"  object_name: {parsed.object_name!r}")
sem = [(a.name, a.key, a.analysis_scope) for a in parsed.semantic_attributes]
qual = [(a.name, a.key) for a in parsed.quality_attributes]
neg = [(a.name, a.key) for a in parsed.negative_attributes]
print(f"  semantic_attrs: {sem}")
print(f"  quality_attrs: {qual}")
print(f"  negative_attrs: {neg}")
pn = any(a.name == "Pure Negative" for a in parsed.negative_attributes)
print(f"  has 'Pure Negative' exact match: {pn}")
print()

print("=== Compiled PipelinePlan ===")
print(f"  steps ({len(plan.steps)}):")
for s in plan.steps:
    print(f"    [{s.order}] {s.step:12s}  model={s.model_id:20s}  flow={s.data_flow.value:10s}  per_candidate={str(s.per_candidate):5s}  params={s.params}")
print()
print(f"  early_exit_rules ({len(plan.early_exit_rules)}):")
for r in plan.early_exit_rules:
    print(f"    condition={r.condition!r}  reason={r.reason!r}")
print()
print(f"  skip_conditions ({len(plan.skip_conditions)}):")
for c in plan.skip_conditions:
    print(f"    step={c.step!r}  condition={c.condition!r}")
