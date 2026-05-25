"""
Analyze every attribute in the fixture template and define its correct execution.
This is the specification that the Planner should produce.
"""
import json, sys
sys.path.insert(0, ".")

fixture = json.loads(open("test/fixtures/request_151049_1tasks.json", encoding="utf-8").read())
template = fixture["template"]["objects"][0]

print("=" * 80)
print("FIXTURE ATTRIBUTE ANALYSIS  —  request_151049_1tasks.json")
print("=" * 80)
print(f"\nobject_name: {template['name']!r}")
print(f"description: {template['description'][:100]}...")
print()

# ── Semantic attributes ──
print("─" * 80)
print("SEMANTIC ATTRIBUTES")
print("─" * 80)
for a in template["attributes"]:
    print(f"\n  [{a['name']}]  type={a['type']}  options={a['options']}")
    print(f"    description: {a['description'][:120]}...")

# ── Quality attributes ──
print("\n")
print("─" * 80)
print("QUALITY ATTRIBUTES")
print("─" * 80)
for a in template.get("quality", {}).get("attributes", []):
    print(f"\n  [{a['name']}]  type={a['type']}")
    print(f"    description: {a['description'][:120]}...")

# ── Negative attributes ──
print("\n")
print("─" * 80)
print("NEGATIVE ATTRIBUTES")
print("─" * 80)
for a in template.get("negative", {}).get("attributes", []):
    print(f"\n  [{a['name']}]  type={a['type']}")
    print(f"    description: {a['description'][:120]}...")
