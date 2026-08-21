from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from .conditions import (
    ConditionModel,
    classify_condition_equation,
    summarize_condition_model,
)
from .domains import DomainGeometry


@dataclass(frozen=True)
class ConditionIssue:
    code: str
    severity: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionAnalysis:
    ok: bool
    issues: tuple[ConditionIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _condition_orders(conditions, *, time_var=None, spatial_vars=()):
    counts: dict[tuple[Any, Any], set[int]] = {}
    for cond in conditions:
        kind = classify_condition_equation(
            cond,
            time_variable=time_var,
            spatial_variables=spatial_vars,
        )
        key = (cond.variable, cond.location)
        order = sum(cond.derivative_multiindex or ())
        counts.setdefault(key, set()).add(order)
        if kind == "velocity":
            counts[key].add(1)
        elif kind == "acceleration":
            counts[key].add(2)
    return counts


def _initial_slice_issues(model, summary):
    issues = []
    slices = tuple(summary.get("time_slices", ()))
    if len(slices) > 1:
        issues.append(
            ConditionIssue(
                "multiple_initial_slices",
                "warning",
                "Initial data is posed on multiple constant-time slices.",
                {"time_slices": slices},
            )
        )
    return issues, slices


def _initial_order_issues(model, time_var, spatial_vars, pde_order, slices):
    if pde_order is None or time_var is None or pde_order < 1:
        return []
    if not model.initial_conditions or not slices:
        return []

    orders = _condition_orders(
        model.initial_conditions,
        time_var=time_var,
        spatial_vars=spatial_vars,
    )
    first = slices[0]
    provided = set()
    for (var, loc), values in orders.items():
        if var == time_var and sp.simplify(loc - first) == 0:
            provided |= set(values)
    provided.add(0)
    required = set(range(min(max(pde_order, 1), 3)))
    if required.issubset(provided):
        return []
    return [
        ConditionIssue(
            "initial_order_gap",
            "warning",
            "Initial data may be insufficient for the PDE order.",
            {
                "required_orders": sorted(required),
                "provided_orders": sorted(provided),
            },
        )
    ]


def _geometry_issues(model, geometry, spatial_vars):
    if geometry is None:
        return []
    issues = []
    bcs = tuple(model.boundary_conditions)
    if geometry.kind == "interval":
        locs = {
            cond.location
            for cond in bcs
            if cond.variable in spatial_vars and cond.location is not None
        }
        if bcs and len(locs) < 2:
            issues.append(
                ConditionIssue(
                    "interval_boundary_incomplete",
                    "warning",
                    "Interval geometry usually expects two boundary locations.",
                    {"locations": tuple(sorted(locs, key=sp.default_sort_key))},
                )
            )
    if geometry.kind == "rectangle":
        sides = {
            (str(cond.variable), cond.location)
            for cond in bcs
            if cond.location is not None
        }
        if bcs and len(sides) < 4:
            issues.append(
                ConditionIssue(
                    "rectangle_boundary_incomplete",
                    "warning",
                    "Rectangle geometry usually expects four boundary components.",
                    {
                        "sides": tuple(
                            sorted(
                                sides,
                                key=lambda item: (
                                    item[0],
                                    sp.default_sort_key(item[1]),
                                ),
                            )
                        )
                    },
                )
            )
    elif len(spatial_vars) == 2 and bcs:
        sides = {
            (str(cond.variable), cond.location)
            for cond in bcs
            if cond.variable in spatial_vars and cond.location is not None
        }
        axes = {
            cond.variable
            for cond in bcs
            if cond.variable in spatial_vars and cond.location is not None
        }
        if sides and len(axes) < 2:
            issues.append(
                ConditionIssue(
                    "rectangle_boundary_incomplete",
                    "warning",
                    "Two-dimensional spatial boundary data does not cover all rectangle boundary components.",
                    {
                        "sides": tuple(
                            sorted(
                                sides,
                                key=lambda item: (
                                    item[0],
                                    sp.default_sort_key(item[1]),
                                ),
                            )
                        )
                    },
                )
            )
    return issues


def _duplicate_issues(model, time_var, spatial_vars):
    issues = []
    seen = set()
    conditions = model.boundary_conditions + model.initial_conditions
    for cond in conditions:
        kind = classify_condition_equation(
            cond,
            time_variable=time_var,
            spatial_variables=spatial_vars,
        )
        key = (kind, cond.variable, cond.location)
        if key in seen:
            issues.append(
                ConditionIssue(
                    "duplicate_condition_slot",
                    "warning",
                    "Multiple conditions occupy the same canonical slot.",
                    {
                        "kind": kind,
                        "variable": cond.variable,
                        "location": cond.location,
                    },
                )
            )
        seen.add(key)
    return issues


def _special_condition_issues(model):
    issues = []
    if model.mixed_conditions:
        issues.append(
            ConditionIssue(
                "mixed_conditions_present",
                "info",
                "Mixed conditions are present and may require specialized handling.",
                {"count": len(model.mixed_conditions)},
            )
        )
    if model.event_conditions:
        issues.append(
            ConditionIssue(
                "event_conditions_present",
                "info",
                "Event-like conditions are present.",
                {"count": len(model.event_conditions)},
            )
        )
    return issues


def _system_data_issues(model, dependent_variables):
    size = len(dependent_variables) if dependent_variables else 1
    if size <= 1 or not model.initial_conditions:
        return []
    if len(model.initial_conditions) >= size:
        return []
    return [
        ConditionIssue(
            "system_initial_data_sparse",
            "warning",
            "System PDE may not have initial data for all components.",
            {"system_size": size, "initial_count": len(model.initial_conditions)},
        )
    ]


def analyze_conditions(
    model: ConditionModel,
    geometry: DomainGeometry | None = None,
    *,
    pde_order: int | None = None,
    dependent_variables: tuple[Any, ...] = (),
) -> ConditionAnalysis:
    """Analyze condition sufficiency, duplication, and domain coverage."""
    indep = tuple(model.independent_variables)
    time_var = model.metadata.get("time_variable")
    if time_var is None and len(indep) >= 2:
        time_var = indep[-1]
    spatial_vars = (
        tuple(v for v in indep if v != time_var) if time_var is not None else indep
    )
    summary = summarize_condition_model(model)

    issues, slices = _initial_slice_issues(model, summary)
    issues.extend(
        _initial_order_issues(
            model,
            time_var,
            spatial_vars,
            pde_order,
            slices,
        )
    )
    issues.extend(_geometry_issues(model, geometry, spatial_vars))
    issues.extend(_duplicate_issues(model, time_var, spatial_vars))
    issues.extend(_special_condition_issues(model))
    issues.extend(_system_data_issues(model, dependent_variables))

    ok = not any(issue.severity == "error" for issue in issues)
    return ConditionAnalysis(
        ok=ok,
        issues=tuple(issues),
        metadata={
            "summary": summary,
            "geometry": getattr(geometry, "kind", None),
        },
    )


__all__ = ["ConditionIssue", "ConditionAnalysis", "analyze_conditions"]
