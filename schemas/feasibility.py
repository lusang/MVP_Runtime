"""
Feasibility rules — maps quality visibility scores to per-attribute gating decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemas.template_spec import ParsedTaskSpec

QUALITY_SCORE: dict[str, dict[str, float]] = {
    "occlusion": {"none": 0.0, "partial": 0.5, "heavy": 1.0},
    "blur":      {"clear": 0.0, "slight": 0.5, "heavy": 1.0},
    "lighting":  {"normal": 0.0, "dim": 0.5, "harsh": 1.0},
}


@dataclass
class FeasibilityRule:
    """Per-attribute quality thresholds. An attribute is feasible only when ALL thresholds are met."""

    attribute_key: str
    max_occlusion: float = 1.0
    max_blur: float = 1.0
    max_lighting_issue: float = 1.0

    def assess(self, visibility: dict[str, Any], metrics: dict[str, Any] | None = None) -> bool | None:
        """Return True if feasible, False if infeasible, None if unknown.

        Prefers continuous metrics over categorical labels for numeric precision.
        """
        metrics = metrics or {}
        any_data = False

        for quality_name, attr_name in (
            ("occlusion", "max_occlusion"),
            ("blur", "max_blur"),
            ("lighting", "max_lighting_issue"),
        ):
            score = _resolve_quality_score(quality_name, visibility, metrics)
            if score is None:
                continue
            any_data = True
            max_allowed = getattr(self, attr_name, 1.0)
            if score > max_allowed:
                return False

        return True if any_data else None


def _resolve_quality_score(
    quality_name: str,
    visibility: dict[str, Any],
    metrics: dict[str, Any],
) -> float | None:
    """Resolve a [0, 1] quality score — metrics first, then categorical labels."""
    # Prefer continuous metrics
    metric_map = {
        "occlusion": "edge_density",
        "blur": "laplacian_var",
        "lighting": "histogram_mean",
    }
    metric_key = metric_map.get(quality_name)
    if metric_key and metric_key in metrics:
        raw = float(metrics[metric_key])
        if quality_name == "occlusion":
            # edge_density: higher → less occlusion. Invert: 1 - normalized density
            # Normal range: 0.0 (heavy occlusion) to 0.12+ (none)
            return max(0.0, min(1.0, 1.0 - raw / 0.12))
        elif quality_name == "blur":
            # laplacian_var: lower → more blurry
            # >150 = clear(0), >50 = slight(0.5), <=50 = heavy(1)
            if raw > 150:
                return 0.0
            elif raw > 50:
                return max(0.0, min(1.0, 1.0 - (raw - 50) / 100.0))
            else:
                return max(0.0, min(1.0, 1.0 - raw / 50.0))
        elif quality_name == "lighting":
            # histogram_mean: deviation from mid-gray (127)
            # 0 deviation = normal(0), 255 deviation = harsh(1)
            return max(0.0, min(1.0, abs(raw - 127) / 127.0))

    # Fallback to categorical label
    entry = visibility.get(quality_name, {})
    if isinstance(entry, dict):
        categorical = str(entry.get("value", "")).lower()
        return QUALITY_SCORE.get(quality_name, {}).get(categorical)

    return None


DEFAULT_FEASIBILITY_RULES: dict[str, FeasibilityRule] = {
    "brand_logo":    FeasibilityRule("brand_logo",    max_occlusion=0.5, max_blur=0.5),
    "package_form":  FeasibilityRule("package_form",  max_occlusion=0.8, max_blur=0.7),
    "size_category": FeasibilityRule("size_category", max_occlusion=0.9, max_blur=0.8),
    "__default__":   FeasibilityRule("__default__",   max_occlusion=0.8, max_blur=0.7),
}


def build_feasibility_rules(parsed: ParsedTaskSpec) -> dict[str, FeasibilityRule]:
    """Merge default rules with template-level quality_requirements overrides."""
    rules: dict[str, FeasibilityRule] = dict(DEFAULT_FEASIBILITY_RULES)

    for attr in parsed.semantic_attributes:
        if not attr.enabled:
            continue
        override = attr.quality_requirements
        if override and isinstance(override, dict):
            base = rules.get(attr.key, rules.get("__default__", FeasibilityRule(attr.key)))
            rules[attr.key] = FeasibilityRule(
                attribute_key=attr.key,
                max_occlusion=float(override.get("max_occlusion", base.max_occlusion)),
                max_blur=float(override.get("max_blur", base.max_blur)),
                max_lighting_issue=float(override.get("max_lighting_issue", base.max_lighting_issue)),
            )

    return rules
