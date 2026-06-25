"""YAML-configurable pluggable optimization objective.

Load an objective via `load_objective(path)` and call `obj.score(metrics)` to
get a scalar (higher-is-better for `direction: maximize`, lower-is-better for
`minimize`). The optimizer converts as needed.

Objective configs under `config/objectives/*.yaml` drive default behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

OBJECTIVES_DIR = Path(__file__).resolve().parent.parent / "config" / "objectives"


@dataclass
class StressTarget:
    vm_target_mpa: float = 200.0
    also_evaluate_at_mpa: List[float] = field(default_factory=list)
    field: str = "avg_von_mises_MPa"
    solver_tolerance_pct: float = 5.0
    max_iters: int = 5
    seed_disp_mm: float = 1.0
    fallback: str = "penalty"  # "penalty" or "skip"
    penalty_weight: float = 1e-3


@dataclass
class ObjectiveTerm:
    metric: str
    weight: float
    target_value: Optional[float] = None
    normalize_by: Optional[float] = None


@dataclass
class Objective:
    name: str
    direction: str  # "minimize" or "maximize"
    terms: List[ObjectiveTerm]
    stress_target: Optional[StressTarget] = None

    def score(self, metrics: Dict[str, Any]) -> float:
        total = 0.0
        for t in self.terms:
            v = metrics.get(t.metric)
            if v is None:
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if t.target_value is not None:
                val = abs(val - float(t.target_value))
            if t.normalize_by:
                val = val / float(t.normalize_by)
            total += float(t.weight) * val

        if self.stress_target and self.stress_target.fallback == "penalty":
            vm = metrics.get(self.stress_target.field)
            if vm is not None:
                try:
                    vm_f = float(vm)
                    delta = (vm_f - self.stress_target.vm_target_mpa) / max(
                        self.stress_target.vm_target_mpa, 1e-6
                    )
                    pen = self.stress_target.penalty_weight * (delta * delta)
                    total = total - pen if self.direction == "maximize" else total + pen
                except (TypeError, ValueError):
                    pass
        return float(total)

    def better(self, a: float, b: float) -> bool:
        return a > b if self.direction == "maximize" else a < b


def load_objective(path: Path) -> Objective:
    raw = _load_yaml_or_json(Path(path))
    terms = [ObjectiveTerm(**t) for t in raw.get("terms", [])]
    st_raw = raw.get("stress_target")
    st = StressTarget(**st_raw) if isinstance(st_raw, dict) else None
    return Objective(
        name=raw["name"],
        direction=raw.get("direction", "maximize"),
        terms=terms,
        stress_target=st,
    )


def load_builtin(name: str) -> Objective:
    for ext in (".yaml", ".yml", ".json"):
        candidate = OBJECTIVES_DIR / f"{name}{ext}"
        if candidate.exists():
            return load_objective(candidate)
    raise FileNotFoundError(f"No builtin objective named '{name}' in {OBJECTIVES_DIR}")


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except ImportError:
            return _naive_yaml(text)
    return json.loads(text)


def _naive_yaml(text: str) -> Dict[str, Any]:
    """Extremely small YAML subset: supports the shapes we use ourselves.

    Only honors: top-level `key: value`, nested one level under `key:`,
    and `- key: value` list items. Good enough for our own configs without
    pulling in a PyYAML dependency at minimum install.
    """

    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    out: Dict[str, Any] = {}
    stack: List[Any] = [out]
    indents = [0]
    for raw in lines:
        stripped = raw.rstrip()
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped.strip()
        while indent < indents[-1]:
            stack.pop()
            indents.pop()
        parent = stack[-1]
        if content.startswith("- "):
            item_str = content[2:].strip()
            if isinstance(parent, list):
                item = _parse_scalar_or_map(item_str)
                parent.append(item)
                if isinstance(item, dict):
                    stack.append(item)
                    indents.append(indent + 2)
        elif ":" in content:
            key, val = content.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val == "":
                # Could be a map or a list; peek ahead by using the next line.
                # Default to map; downgraded to list if first child is "- ".
                next_child: Any = {}
                parent[key] = next_child
                stack.append(next_child)
                indents.append(indent + 2)
            elif val.startswith("[") and val.endswith("]"):
                parent[key] = [_coerce_scalar(s.strip()) for s in val[1:-1].split(",") if s.strip()]
            else:
                parent[key] = _coerce_scalar(val)
    return out


def _parse_scalar_or_map(s: str) -> Any:
    if ":" in s:
        k, v = s.split(":", 1)
        return {k.strip(): _coerce_scalar(v.strip())}
    return _coerce_scalar(s)


def _coerce_scalar(s: str) -> Any:
    if s == "":
        return None
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s
