from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from .conditions import ConditionModel, classify_condition_equation, summarize_condition_model
from .domains import DomainGeometry
from .results import EigenfunctionExpansionResult, SeriesPDEResult
from .separation_general import separate_product_pde
from .sturm_liouville import SturmLiouvilleProblem, solve_regular_constant_sturm_liouville


@dataclass(frozen=True)
class SeparableGeometryPlan:
    geometry_kind: str
    boundary_family: str
    eigenbasis: str
    spatial_variables: tuple[sp.Symbol, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def build_separable_geometry_plan(
    geometry: DomainGeometry | None,
    conditions: ConditionModel | None,
    *,
    family: str | None = None,
) -> SeparableGeometryPlan | None:
    if geometry is None or conditions is None:
        return None
    summary = summarize_condition_model(conditions)
    bc_kinds = set(summary.get("boundary_kinds", ()))
    spatial_variables = tuple(
        conditions.independent_variables[:-1]
        if len(conditions.independent_variables) >= 2
        else conditions.independent_variables
    )
    kind = geometry.kind
    if kind == "interval":
        if bc_kinds == {"dirichlet"}:
            return SeparableGeometryPlan(
                kind, "dirichlet", "sine", spatial_variables, {"series_family": family}
            )
        if bc_kinds == {"neumann"}:
            return SeparableGeometryPlan(
                kind, "neumann", "cosine", spatial_variables, {"series_family": family}
            )
        if "robin" in bc_kinds:
            return SeparableGeometryPlan(
                kind, "robin", "sturm_liouville_robin", spatial_variables, {"series_family": family}
            )
    if kind == "rectangle":
        if not bc_kinds or bc_kinds == {"dirichlet"}:
            return SeparableGeometryPlan(
                kind, "dirichlet", "tensor_sine", spatial_variables, {"series_family": family}
            )
        if bc_kinds == {"neumann"}:
            return SeparableGeometryPlan(
                kind, "neumann", "tensor_cosine", spatial_variables, {"series_family": family}
            )
        if "robin" in bc_kinds:
            return SeparableGeometryPlan(
                kind,
                "robin",
                "tensor_sturm_liouville",
                spatial_variables,
                {"series_family": family},
            )
    if kind == "disk":
        return SeparableGeometryPlan(
            kind, "radial_angular", "bessel_fourier", spatial_variables, {"series_family": family}
        )
    if kind == "polar_annulus":
        return SeparableGeometryPlan(
            kind,
            "radial_angular",
            "bessel_annulus_fourier",
            spatial_variables,
            {"series_family": family},
        )
    return None


def _time_var(model: ConditionModel | None):
    if model is None or len(model.independent_variables) < 2:
        return None
    return model.independent_variables[-1]


def _spatial_vars(model: ConditionModel | None):
    if model is None:
        return ()
    return tuple(
        model.independent_variables[:-1]
        if len(model.independent_variables) >= 2
        else model.independent_variables
    )


def _condition_rhs_for_kind(model: ConditionModel | None, kind: str):
    if model is None:
        return None
    spatial = _spatial_vars(model)
    tvar = _time_var(model)
    for cond in model.initial_conditions:
        if classify_condition_equation(cond, time_variable=tvar, spatial_variables=spatial) == kind:
            return cond.equation.rhs
    return None


def _boundary_rhs(
    model: ConditionModel | None, *, variable=None, location=None, kind: str | None = None
):
    if model is None:
        return None
    spatial = _spatial_vars(model)
    tvar = _time_var(model)
    for cond in model.boundary_conditions:
        if variable is not None and cond.variable != variable:
            continue
        if location is not None:
            try:
                if cond.location is None or sp.simplify(cond.location - location) != 0:
                    continue
            except Exception:
                if cond.location != location:
                    continue
        ck = classify_condition_equation(cond, time_variable=tvar, spatial_variables=spatial)
        if kind is not None and ck != kind:
            continue
        return cond.equation.rhs
    return None


def _geometry_interval_length(geometry: DomainGeometry | None):
    if geometry is None:
        return None
    for ext in getattr(geometry, "extents", {}).values():
        if isinstance(ext, tuple) and len(ext) == 2:
            try:
                return sp.simplify(ext[1] - ext[0])
            except Exception:
                return ext[1]
    return None


def _rectangle_lengths(geometry: DomainGeometry | None):
    if geometry is None:
        return None, None
    xext = getattr(geometry, "extents", {}).get("x")
    yext = getattr(geometry, "extents", {}).get("y")
    xl = sp.simplify(xext[1] - xext[0]) if isinstance(xext, tuple) and len(xext) == 2 else None
    yl = sp.simplify(yext[1] - yext[0]) if isinstance(yext, tuple) and len(yext) == 2 else None
    return xl, yl


def _as_series_result(solution, plan, *, eigen_data=None, native=True):
    metadata = {
        "plan": plan,
        "separation_plan": plan,
        "native_structured_execution": native,
        "planner_method": "separation_framework",
    }
    method = getattr(solution, "method", "separation_framework")
    if isinstance(solution, sp.Sum):
        return EigenfunctionExpansionResult(
            method=method,
            solution=solution,
            metadata=metadata,
            series_terms=None,
            eigen_data=eigen_data,
        )
    return SeriesPDEResult(
        method=method,
        solution=getattr(solution, "solution", solution),
        metadata=metadata,
        series_terms=None,
    )


def _regular_interval_spectrum(x, L, boundary_family):
    X = sp.Function("X")(x)
    try:
        return solve_regular_constant_sturm_liouville(
            SturmLiouvilleProblem(x, X, 1, 0, 1, (0, L), boundary_family, boundary_family)
        )
    except Exception:
        return None


def execute_separation_plan(
    problem, classical_mod, *, plan: SeparableGeometryPlan | None = None, **kwargs
):
    """Execute a structured separation/series plan using ConditionModel and DomainGeometry directly."""
    plan = (
        plan
        or (getattr(problem.canonical_representation, "details", {}) or {}).get("separation_plan")
        or problem.details.get("separation_plan")
    )
    if plan is None:
        # lightweight fallback synthesis from canonical/domain/BC metadata
        geom = problem.details.get("domain_geometry")
        if geom is not None and geom.kind == "interval":
            bcs = problem.bcs if isinstance(problem.bcs, dict) else {}
            btype = bcs.get("type")
            if btype == "dirichlet_homogeneous_interval":
                plan = SeparableGeometryPlan(
                    "interval",
                    "dirichlet",
                    "sine",
                    (problem.indep_vars[0],),
                    {"series_family": getattr(problem.profile, "canonical_family", None)},
                )
            elif btype == "neumann_homogeneous_interval":
                plan = SeparableGeometryPlan(
                    "interval",
                    "neumann",
                    "cosine",
                    (problem.indep_vars[0],),
                    {"series_family": getattr(problem.profile, "canonical_family", None)},
                )
        if plan is None:
            raise NotImplementedError("No structured separation plan available.")
    cond_model = problem.details.get("condition_model") or (
        getattr(problem.canonical_representation, "details", {}) or {}
    ).get("condition_model")
    geometry = problem.details.get("domain_geometry") or (
        getattr(problem.canonical_representation, "details", {}) or {}
    ).get("domain_geometry")
    family = getattr(problem.profile, "canonical_family", None)
    uexpr = problem.dep_function
    vars_ = problem.indep_vars

    if plan.geometry_kind == "interval":
        x = plan.spatial_variables[0] if plan.spatial_variables else vars_[0]
        t = _time_var(cond_model) or (vars_[1] if len(vars_) > 1 else None)
        L = _geometry_interval_length(geometry) or kwargs.get("length", sp.pi)
        profile = _condition_rhs_for_kind(cond_model, "profile")
        velocity = _condition_rhs_for_kind(cond_model, "velocity")
        if profile is None and isinstance(getattr(problem, "ics", None), dict):
            profile = problem.ics.get("initial_profile", problem.ics.get("initial_displacement"))
        if velocity is None and isinstance(getattr(problem, "ics", None), dict):
            velocity = problem.ics.get(
                "initial_velocity", problem.ics.get("initial_time_derivative")
            )
        if family == "heat_like" and profile is not None:
            if plan.boundary_family == "dirichlet":
                return _as_series_result(
                    classical_mod.solve_heat_equation_1d_dirichlet_series(
                        uexpr,
                        x=x,
                        t=t,
                        diffusivity=kwargs.get("diffusivity", 1),
                        length=L,
                        initial_profile=profile,
                        terms=kwargs.get("terms", 6),
                    ),
                    plan,
                    eigen_data={
                        "basis": "sine",
                        "sturm_liouville": _regular_interval_spectrum(x, L, "dirichlet"),
                    },
                )
            if plan.boundary_family == "neumann":
                return _as_series_result(
                    classical_mod.solve_heat_equation_1d_neumann_series(
                        uexpr,
                        x=x,
                        t=t,
                        diffusivity=kwargs.get("diffusivity", 1),
                        length=L,
                        initial_profile=profile,
                        terms=kwargs.get("terms", 6),
                    ),
                    plan,
                    eigen_data={
                        "basis": "cosine",
                        "sturm_liouville": _regular_interval_spectrum(x, L, "neumann"),
                    },
                )
            if plan.boundary_family == "robin":
                left = _boundary_rhs(cond_model, variable=x, location=0, kind="robin")
                right = _boundary_rhs(cond_model, variable=x, location=L, kind="robin")
                return _as_series_result(
                    classical_mod.solve_heat_equation_1d_robin_series(
                        uexpr,
                        x=x,
                        t=t,
                        diffusivity=kwargs.get("diffusivity", 1),
                        length=L,
                        initial_profile=profile,
                        h0=kwargs.get("h0", left if left is not None else 1),
                        hL=kwargs.get("hL", right if right is not None else 1),
                        terms=kwargs.get("terms", 6),
                    ),
                    plan,
                    eigen_data={"basis": "sturm_liouville_robin"},
                )
        if family == "wave_like" and profile is not None:
            return _as_series_result(
                classical_mod.solve_wave_equation_1d_dirichlet_series(
                    uexpr,
                    x=x,
                    t=t,
                    wave_speed=kwargs.get("wave_speed", 1),
                    length=L,
                    initial_displacement=profile,
                    initial_velocity=velocity,
                    terms=kwargs.get("terms", 6),
                ),
                plan,
                eigen_data={
                    "basis": "sine",
                    "sturm_liouville": _regular_interval_spectrum(x, L, "dirichlet"),
                },
            )

    if plan.geometry_kind == "rectangle" and family in {
        "laplace_like",
        "helmholtz_like",
        "diffusion_reaction",
        "reaction_diffusion_like",
    }:
        x = plan.spatial_variables[0] if len(plan.spatial_variables) > 0 else vars_[0]
        y = plan.spatial_variables[1] if len(plan.spatial_variables) > 1 else vars_[1]
        xl, yl = _rectangle_lengths(geometry)
        top = _boundary_rhs(
            cond_model,
            variable=y,
            location=(getattr(geometry, "extents", {}).get("y") or (None, None))[1],
            kind="dirichlet",
        )
        return _as_series_result(
            classical_mod.solve_rectangle_dirichlet_laplace_series(
                uexpr,
                x=x,
                y=y,
                x_length=xl or kwargs.get("x_length", sp.pi),
                y_length=yl or kwargs.get("y_length", sp.pi),
                boundary_top=top,
                terms=kwargs.get("terms", 6),
            ),
            plan,
            eigen_data={"basis": "tensor_sine"},
        )

    # Generic multiplicative separation fallback derived from the supplied PDE.
    try:
        generic = separate_product_pde(problem.equation, uexpr, vars_)
        return SeriesPDEResult(
            method="general_product_separation",
            solution=generic.ansatz,
            metadata={
                "plan": plan,
                "separation_result": generic,
                "native_structured_execution": False,
                "planner_method": "separation_framework",
            },
            series_terms=None,
        )
    except (ValueError, NotImplementedError):
        return _as_series_result(
            classical_mod.separate_variables_structured(
                problem.equation, uexpr, vars_, assumptions=problem.assumptions, bcs=problem.bcs
            ).ansatz,
            plan,
            native=False,
        )


__all__ = ["SeparableGeometryPlan", "build_separable_geometry_plan", "execute_separation_plan"]
