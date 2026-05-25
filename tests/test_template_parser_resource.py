"""Tests for resource/Template.json parsing."""

import json
from pathlib import Path

import pytest

from runtime.template_parser import TemplateParser

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "resource" / "Template.json"


def test_parse_resource_template_handlers_and_counts():
    raw = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    parsed = TemplateParser().parse(raw)

    assert parsed.object_name == "Package"
    assert len(parsed.semantic_attributes) == 3
    assert len(parsed.quality_attributes) == 3
    assert len(parsed.negative_attributes) == 4

    assert parsed.semantic_attributes[0].handler == "gemini"
    assert parsed.semantic_attributes[0].scope == "semantic"
    assert parsed.semantic_attributes[0].name == "package_form"

    assert parsed.quality_attributes[0].handler == "opencv_quality"
    assert parsed.quality_attributes[0].scope == "quality"

    assert parsed.negative_attributes[0].handler == "gemini_negative"
    assert parsed.negative_attributes[0].scope == "negative"

    assert len(parsed.all_attribute_slots) == 10


def test_parse_requires_objects_array():
    with pytest.raises(ValueError, match="objects"):
        TemplateParser().parse({})
