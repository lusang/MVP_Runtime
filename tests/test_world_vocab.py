"""Tests for YOLO-World vocabulary builder."""

from models.object_target import ObjectTarget
from models.world_vocab import world_classes_from_target


def test_world_vocab_includes_name_and_include_bullets():
    target = ObjectTarget(
        name="Package",
        description="container",
        include="- 快递纸箱 - 文件信封",
        exclude="",
        geometry="bbox",
    )
    classes = world_classes_from_target(target, max_classes=20)
    assert "Package" in classes
    assert "快递纸箱" in classes
    assert "文件信封" in classes
    assert "package" in classes
