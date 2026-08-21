from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from .conditions import ConditionModel, classify_condition_equation
from .domains import DomainGeometry


@dataclass(frozen=True)
class BoundaryComponent:
    name: str
    variable: sp.Symbol | None
    location: sp.Expr | None
    geometry_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryConditionBinding:
    component: BoundaryComponent
    kind: str
    equation: sp.Equality
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryModel:
    geometry: DomainGeometry | None
    components: tuple[BoundaryComponent, ...] = ()
    bindings: tuple[BoundaryConditionBinding, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def build_boundary_model(
    geometry: DomainGeometry | None, conditions: ConditionModel | None
) -> BoundaryModel | None:
    if geometry is None or conditions is None:
        return None
    spatial_vars = tuple(
        conditions.independent_variables[:-1]
        if len(conditions.independent_variables) >= 2
        else conditions.independent_variables
    )
    comps = []
    if geometry.extents:
        for key, ext in geometry.extents.items():
            if not isinstance(ext, tuple) or len(ext) != 2:
                continue
            var = next((v for v in spatial_vars if str(v) == key or str(v) == key[0]), None)
            comps.append(BoundaryComponent(f"{key}_lower", var, ext[0], geometry.kind))
            comps.append(BoundaryComponent(f"{key}_upper", var, ext[1], geometry.kind))
    bindings = []
    for cond in getattr(conditions, "boundary_conditions", ()):
        kind = classify_condition_equation(
            cond,
            time_variable=conditions.independent_variables[-1]
            if len(conditions.independent_variables) >= 2
            else None,
            spatial_variables=spatial_vars,
        )
        comp = next(
            (
                c
                for c in comps
                if c.variable == cond.variable and sp.simplify(c.location - cond.location) == 0
            ),
            None,
        )
        if comp is None:
            comp = BoundaryComponent(
                f"boundary_{len(comps) + 1}", cond.variable, cond.location, geometry.kind
            )
            comps.append(comp)
        bindings.append(BoundaryConditionBinding(comp, kind, cond.equation, dict(cond.metadata)))
    return BoundaryModel(geometry, tuple(comps), tuple(bindings), {"geometry_kind": geometry.kind})


__all__ = ["BoundaryComponent", "BoundaryConditionBinding", "BoundaryModel", "build_boundary_model"]
