"""
DAG Visualizer — builds canonical DAG snapshots from PipelinePlan.

Produces:
  - dag_snapshot dict (nodes, edges, rules) for storage / debugging
  - graph_hash (sha256 prefix) for topology versioning
  - text rendering (ASCII) for console / log output
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from schemas.pipeline_plan import PipelinePlan


def build_dag_snapshot(plan: PipelinePlan) -> dict[str, Any]:
    """Build a canonical DAG snapshot from a PipelinePlan.

    The snapshot is deterministic (same plan → same snapshot → same hash).
    It contains everything needed to reconstruct the execution topology
    without access to the original template.
    """
    nodes = []
    for s in plan.steps:
        node: dict[str, Any] = {
            "step": s.step,
            "order": s.order,
            "model_id": s.model_id,
            "data_flow": s.data_flow.value,
        }
        if s.scope:
            node["scope"] = s.scope
        if s.params.get("attribute_keys"):
            node["attributes"] = s.params["attribute_keys"]
        nodes.append(node)

    # Edges: sequential ordering + early_exit / skip annotations
    edges = []
    for i in range(len(nodes) - 1):
        edge: dict[str, Any] = {
            "from": nodes[i]["step"],
            "to": nodes[i + 1]["step"],
            "type": "data",
        }
        edges.append(edge)

    # Early-exit rules as "virtual edges" from their source step
    early_exit_sources: set[str] = set()
    for rule in plan.early_exit_rules:
        early_exit_sources.add(rule.condition.split(".")[0] if "." in rule.condition else "scene_negative")
    for source in early_exit_sources:
        edges.append({
            "from": source,
            "to": "__END__",
            "type": "early_exit",
            "condition": next(
                (r.condition for r in plan.early_exit_rules),
                "",
            ),
        })

    skip_rules = [
        {"step": c.step, "condition": c.condition, "reason": c.reason}
        for c in plan.skip_conditions
    ]
    early_exit_rules = [
        {"condition": r.condition, "reason": r.reason}
        for r in plan.early_exit_rules
    ]

    return {
        "graph_hash": "",  # placeholder — filled by compute_graph_hash
        "plan_id": plan.plan_id,
        "object_name": plan.object_name,
        "planner_version": plan.planner_version,
        "planner_model": plan.planner_model,
        "nodes": nodes,
        "edges": edges,
        "skip_rules": skip_rules,
        "early_exit_rules": early_exit_rules,
    }


def compute_graph_hash(dag: dict[str, Any]) -> str:
    """Compute a deterministic hash for a DAG snapshot.

    The hash excludes fields that change per-run (plan_id, graph_hash itself).
    Same template + same planner → same hash.
    """
    payload = {k: v for k, v in dag.items() if k not in ("graph_hash", "plan_id")}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def render_dag_text(dag: dict[str, Any]) -> str:
    """Render a DAG snapshot as ASCII art for console / log output."""
    lines = []
    nodes = dag.get("nodes", [])
    skip_map = {r["step"]: r for r in dag.get("skip_rules", [])}
    early_map = {r["condition"]: r for r in dag.get("early_exit_rules", [])}

    for i, node in enumerate(nodes):
        step = node["step"]
        model = node["model_id"]
        attrs = node.get("attributes")
        attrs_str = f" ({', '.join(attrs)})" if attrs else ""

        # Node line
        lines.append(f"  [{node['order']}] {step}  [{model}]{attrs_str}")

        # Skip annotation
        if step in skip_map:
            r = skip_map[step]
            lines.append(f"       │  skip_if → {r['condition']}")

        # Early-exit annotation (only on first node)
        if i == 0 and early_map:
            ee = next(iter(early_map.values()))
            lines.append(f"       │  early_exit → {ee['condition']}")

        # Connector (not after last node)
        if i < len(nodes) - 1:
            lines.append(f"       ▼")

    return "\n".join(lines)


def pipeline_summary_text(plan: PipelinePlan, dag: dict[str, Any]) -> str:
    """One-line summary for log output."""
    steps = [f"{s.step}:{s.model_id}" for s in plan.steps]
    return (
        f"PipelinePlan [hash={dag.get('graph_hash', '?')}] "
        f"{len(plan.steps)} steps: {' → '.join(steps)}"
    )
