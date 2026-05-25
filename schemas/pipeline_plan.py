"""
PipelinePlan — structured JSON output from Planner, consumed by StepExecutor.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DataFlow(str, Enum):
    CROP = "crop"
    FULL = "full_image"


class PlanStep(BaseModel):
    """One executable step in the pipeline plan."""

    model_config = ConfigDict(extra="forbid")

    step: str = Field(..., description="detect, nms, verify, attribute, quality, negative, merge")
    model_id: str = Field(..., description="Model from ModelRegistry")
    data_flow: DataFlow = Field(default=DataFlow.CROP)
    order: int = Field(..., ge=0, description="Execution order — sorted ascending")
    per_candidate: bool = Field(default=False)
    scope: str | None = Field(default=None, description="For attribute steps: semantic, quality, negative")
    params: dict[str, Any] = Field(default_factory=dict)


class EarlyExitRule(BaseModel):
    """Condition to stop the entire pipeline early."""

    model_config = ConfigDict(extra="forbid")

    condition: str = Field(..., description="Expression evaluated against context, e.g. len(detections)==0")
    reason: str = Field(default="")


class SkipCondition(BaseModel):
    """Condition to skip a specific step."""

    model_config = ConfigDict(extra="forbid")

    step: str = Field(..., description="Step name to skip")
    condition: str = Field(..., description="Expression evaluated against context, e.g. bbox_conf < 0.3")
    reason: str = Field(default="")


class PipelinePlan(BaseModel):
    """Complete execution plan produced by the Planner."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(..., description="UUID for this plan instance")
    object_name: str = Field(..., description="Target object from template")
    steps: list[PlanStep] = Field(..., min_length=1, description="Ordered list of executable steps")
    early_exit_rules: list[EarlyExitRule] = Field(default_factory=list)
    skip_conditions: list[SkipCondition] = Field(default_factory=list)
    planner_model: str = Field(default="gemini-2.0-flash")
    planner_version: str = Field(default="1.0")
    meta: dict[str, Any] = Field(default_factory=dict)
