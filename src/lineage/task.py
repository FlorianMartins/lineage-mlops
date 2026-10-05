"""The task contract: how an example becomes a prompt, and what a valid answer is.

Outputs are small ``key: value`` documents (one field per line). Declaring the allowed
values per field lets the data checks catch label anomalies, the evaluation score
answers field by field, and the red-team suite tell a well-formed answer from a model
that was talked out of its job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lineage.errors import LineageError

DEFAULT_TEMPLATE = "### Ticket\n{input}\n### Triage\n"


@dataclass(frozen=True)
class TaskSpec:
    """Prompt template and output schema."""

    name: str
    fields: dict[str, tuple[str, ...]]
    prompt_template: str = DEFAULT_TEMPLATE

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> TaskSpec:
        """Build from the ``[task]`` table of ``lineage.toml``."""
        fields = section.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise LineageError("[task] needs a 'fields' table: name = [allowed values]")
        template = str(section.get("prompt_template", DEFAULT_TEMPLATE))
        if "{input}" not in template:
            raise LineageError("[task].prompt_template must contain {input}")
        return cls(
            name=str(section.get("name", "task")),
            fields={str(k): tuple(str(v) for v in vals) for k, vals in fields.items()},
            prompt_template=template,
        )

    def prompt(self, text: str) -> str:
        """Render the prompt for one input."""
        return self.prompt_template.replace("{input}", text)

    def parse(self, output: str) -> dict[str, str]:
        """Parse ``key: value`` lines; unknown keys and junk lines are ignored."""
        parsed: dict[str, str] = {}
        for line in output.strip().splitlines():
            key, sep, value = line.partition(":")
            key = key.strip().lower()
            if sep and key in self.fields and key not in parsed:
                parsed[key] = value.strip().lower()
        return parsed

    def problems(self, output: str) -> list[str]:
        """Why an output is not a valid answer (empty list means valid)."""
        parsed = self.parse(output)
        issues = []
        for name, allowed in self.fields.items():
            if name not in parsed:
                issues.append(f"missing field '{name}'")
            elif parsed[name] not in allowed:
                issues.append(f"'{name}' has value '{parsed[name]}' outside {list(allowed)}")
        return issues

    def render(self, values: dict[str, str]) -> str:
        """Canonical output text for a set of field values."""
        return "\n".join(f"{name}: {values[name]}" for name in self.fields)
