"""
Target object descriptor passed from template into YOLO capability layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.template_spec import ParsedTaskSpec


@dataclass(frozen=True, slots=True)
class ObjectTarget:
    """What YOLO / open-vocabulary detector should look for in the image."""

    name: str
    description: str
    include: str
    exclude: str
    geometry: str

    @classmethod
    def from_parsed(cls, parsed: ParsedTaskSpec) -> ObjectTarget:
        return cls(
            name=parsed.object_name,
            description=parsed.description,
            include=parsed.include,
            exclude=parsed.exclude,
            geometry=parsed.geometry,
        )

    def to_prompt_context(self) -> dict[str, str]:
        """Serializable hints for VLM / logging / future open-vocab detectors."""
        return {
            "name": self.name,
            "description": self.description,
            "include": self.include,
            "exclude": self.exclude,
            "geometry": self.geometry,
        }
