# Structured benchmark output — one report per feature run, split into sections.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkSection:
    """One panel in the UI / one block in CLI output."""

    title: str
    body: str
    kind: str = "markdown"  # markdown | code | table


@dataclass
class BenchmarkReport:
    """Full result of running one feature benchmark."""

    feature_id: int
    feature_name: str
    subtitle: str = ""
    sections: list[BenchmarkSection] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_cli(self) -> str:
        lines = [
            "",
            "═" * 78,
            f"  Feature {self.feature_id}: {self.feature_name}",
        ]
        if self.subtitle:
            lines.append(f"  {self.subtitle}")
        lines.append("═" * 78)

        for section in self.sections:
            lines.extend(["", "─" * 78, f"  {section.title}", "─" * 78, ""])
            lines.append(section.body)

        lines.extend(["", "═" * 78, ""])
        return "\n".join(lines)
