from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from .conditions import ConditionModel, classify_condition_equation, summarize_condition_model
from .domains import DomainGeometry
from .results import CanonicalPDERepresentation, TransformPDEResult
from .transform_postprocess import postprocess_transform_result
from .unified_transform import recognize_evolution_pde, solve_unified_transform


@dataclass(frozen=True)
class TransformMethodPlan:
    method: str
    domain: str
    transform_family: str
    required_conditions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def build_transform_method_plan(
    canonical: CanonicalPDERepresentation | None,
    geometry: DomainGeometry | None,
    conditions: ConditionModel | None,
) -> TransformMethodPlan | None:
    if canonical is None or geometry is None:
        return None
    tags = set(canonical.transformability_tags or ())
    summary = summarize_condition_model(conditions) if conditions is not None else {}
    bc_kinds = set(summary.get("boundary_kinds", ()))
    if "integral_transform" not in tags and "constant_coefficients" not in tags:
        return None
    if geometry.kind == "full_line":
        return TransformMethodPlan(
            "structured_transform",
            "whole_line",
            "fourier",
            ("profile",),
            {"geometry_kind": geometry.kind},
        )
    if geometry.kind == "half_line":
        fam = (
            "laplace_sine"
            if "dirichlet" in bc_kinds
            else "laplace_cosine"
            if "neumann" in bc_kinds
            else "laplace"
        )
        return TransformMethodPlan(
            "structured_transform", "half_line", fam, ("profile",), {"geometry_kind": geometry.kind}
        )
    if geometry.kind == "interval" and bc_kinds:
        fam = (
            "finite_sine"
            if bc_kinds == {"dirichlet"}
            else "finite_cosine"
            if bc_kinds == {"neumann"}
            else "sturm_liouville_transform"
        )
        return TransformMethodPlan(
            "structured_transform", "interval", fam, ("profile",), {"geometry_kind": geometry.kind}
        )
    if geometry.kind == "unspecified_spacetime" and "integral_transform" in tags:
        return TransformMethodPlan(
            "structured_transform",
            "unspecified_spacetime",
            "laplace_fourier",
            (),
            {"geometry_kind": geometry.kind},
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


def initial_condition_equation(model: ConditionModel | None):
    if model is None:
        return None
    spatial = _spatial_vars(model)
    tvar = _time_var(model)
    for cond in model.initial_conditions:
        if (
            classify_condition_equation(cond, time_variable=tvar, spatial_variables=spatial)
            == "profile"
        ):
            return cond.equation
    return model.initial_conditions[0].equation if model.initial_conditions else None


def boundary_condition_equations(model: ConditionModel | None):
    if model is None:
        return ()
    return tuple(cond.equation for cond in model.boundary_conditions)


def _as_transform_result(solution, plan, *, wrapped=None, native=True):
    result = TransformPDEResult(
        method=getattr(solution, "method", "structured_transform"),
        solution=getattr(solution, "solution", solution),
        metadata={
            "plan": plan,
            "transform_data": {"family": plan.transform_family, "domain": plan.domain},
            "native_structured_execution": native,
            "wrapped_result": wrapped,
            "planner_method": "structured_transform",
        },
    )
    processed, report = postprocess_transform_result(result)
    md = dict(processed.metadata)
    md["transform_postprocess"] = {
        "changed": report.changed,
        "stages": report.stages,
        "skipped_reason": report.skipped_reason,
    }
    return TransformPDEResult(
        method=processed.method,
        solution=processed.solution,
        classification=processed.classification,
        assumptions=processed.assumptions,
        verification=processed.verification,
        metadata=md,
        reduced_problem=processed.reduced_problem,
        warnings=processed.warnings,
        transform_data=processed.transform_data,
    )


def execute_transform_plan(
    problem, classical_mod=None, *, plan: TransformMethodPlan | None = None, **kwargs
):
    plan = (
        plan
        or (getattr(problem.canonical_representation, "details", {}) or {}).get("transform_plan")
        or problem.details.get("transform_plan")
    )
    if plan is None:
        raise NotImplementedError("No structured transform plan available.")
    classical_mod = classical_mod or __import__("pdesolve.classical_methods", fromlist=["dummy"])
    cond_model = problem.details.get("condition_model") or (
        getattr(problem.canonical_representation, "details", {}) or {}
    ).get("condition_model")
    ic_eq = initial_condition_equation(cond_model)
    bc_eqs = list(boundary_condition_equations(cond_model))
    if (
        ic_eq is None
        and isinstance(getattr(problem, "ics", None), dict)
        and problem.ics.get("initial_profile") is not None
        and len(problem.indep_vars) >= 2
    ):
        x0 = problem.indep_vars[0]
        ic_eq = sp.Eq(problem.dep_function.func(x0, 0), problem.ics.get("initial_profile"))
    if ic_eq is None:
        raise ValueError("structured_transform requires an explicit initial-condition equation.")

    ufunc = problem.dep_function.func
    vars2 = problem.indep_vars[:2]
    x = vars2[0]
    t = vars2[1] if len(vars2) > 1 else None
    profile = ic_eq.rhs

    # Native transform execution paths.
    if plan.domain == "whole_line":
        ev = recognize_evolution_pde(problem.equation, ufunc, vars2)
        if ev is not None and ev.family_name == "heat_like":
            return _as_transform_result(
                classical_mod.solve_heat_equation_1d_fourier_transform(
                    problem.dep_function,
                    x=x,
                    t=t,
                    diffusivity=kwargs.get("diffusivity", 1),
                    initial_profile=profile,
                ),
                plan,
            )
        if ev is not None and ev.family_name == "advection_like":
            coeffs = ev.coeffs
            a_t = sp.sympify(ev.time_coefficient)
            speed = sp.simplify(coeffs.get(1, 0) / a_t) if a_t != 0 else 1
            reaction = sp.simplify(coeffs.get(0, 0) / a_t) if a_t != 0 else 0
            return _as_transform_result(
                classical_mod.solve_advection_equation_1d_fourier_transform(
                    problem.dep_function,
                    x=x,
                    t=t,
                    speed=speed,
                    reaction=reaction,
                    initial_profile=profile,
                ),
                plan,
            )
        return _as_transform_result(
            solve_unified_transform(
                problem.equation,
                ufunc,
                vars2,
                initial_condition=ic_eq,
                boundary_conditions=bc_eqs,
                domain=plan.domain,
            ),
            plan,
            wrapped="unified_transform",
            native=False,
        )

    if plan.domain == "half_line":
        if "sine" in plan.transform_family:
            return _as_transform_result(
                classical_mod.solve_heat_equation_1d_half_line_transform(
                    problem.dep_function,
                    x=x,
                    t=t,
                    diffusivity=kwargs.get("diffusivity", 1),
                    initial_profile=profile,
                    boundary="dirichlet",
                ),
                plan,
            )
        if "cosine" in plan.transform_family:
            return _as_transform_result(
                classical_mod.solve_heat_equation_1d_half_line_transform(
                    problem.dep_function,
                    x=x,
                    t=t,
                    diffusivity=kwargs.get("diffusivity", 1),
                    initial_profile=profile,
                    boundary="neumann",
                ),
                plan,
            )
        return _as_transform_result(
            solve_unified_transform(
                problem.equation,
                ufunc,
                vars2,
                initial_condition=ic_eq,
                boundary_conditions=bc_eqs,
                domain=plan.domain,
            ),
            plan,
            wrapped="unified_transform",
            native=False,
        )

    if plan.domain == "interval":
        L = kwargs.get("length", sp.pi)
        if (
            plan.transform_family == "finite_sine"
            and getattr(problem.profile, "canonical_family", None) == "wave_like"
        ):
            velocity = None
            spatial = _spatial_vars(cond_model)
            tvar = _time_var(cond_model)
            for cond in cond_model.initial_conditions:
                if (
                    classify_condition_equation(cond, time_variable=tvar, spatial_variables=spatial)
                    == "velocity"
                ):
                    velocity = cond.equation.rhs
                    break
            return _as_transform_result(
                classical_mod.solve_wave_equation_1d_laplace_sine_transform_formal(
                    problem.dep_function,
                    x=x,
                    t=t,
                    wave_speed=kwargs.get("wave_speed", 1),
                    length=L,
                    initial_displacement=profile,
                    initial_velocity=velocity,
                ),
                plan,
            )
        if plan.transform_family == "finite_sine":
            return _as_transform_result(
                classical_mod.solve_heat_equation_1d_laplace_transform_formal(
                    problem.dep_function,
                    x=x,
                    t=t,
                    diffusivity=kwargs.get("diffusivity", 1),
                    initial_profile=profile,
                ),
                plan,
            )
        if plan.transform_family == "finite_cosine":
            return _as_transform_result(
                classical_mod.solve_heat_equation_1d_laplace_transform_formal(
                    problem.dep_function,
                    x=x,
                    t=t,
                    diffusivity=kwargs.get("diffusivity", 1),
                    initial_profile=profile,
                ),
                plan,
            )

    if plan.domain == "unspecified_spacetime":
        fam = getattr(problem.profile, "canonical_family", None)
        if fam == "heat_like":
            # With an initial profile and no spatial boundary data, the natural
            # interpretation is the whole-line Cauchy problem.  Preserve the
            # public Fourier-method result identity used by the public API.
            ev = recognize_evolution_pde(problem.equation, ufunc, vars2)
            diffusivity = kwargs.get("diffusivity", 1)
            if ev is not None and ev.family_name == "heat_like":
                a_t = sp.sympify(ev.time_coefficient)
                if a_t != 0:
                    diffusivity = sp.simplify(-ev.coeffs.get(2, 0) / a_t)
            return _as_transform_result(
                classical_mod.solve_heat_equation_1d_fourier_transform(
                    problem.dep_function, x=x, t=t, diffusivity=diffusivity, initial_profile=profile
                ),
                plan,
            )
        if fam == "wave_like":
            return _as_transform_result(
                classical_mod.solve_wave_equation_1d_laplace_transform_formal(
                    problem.dep_function,
                    x=x,
                    t=t,
                    wave_speed=kwargs.get("wave_speed", 1),
                    initial_displacement=profile,
                    initial_velocity=kwargs.get("initial_velocity"),
                ),
                plan,
            )

    res = solve_unified_transform(
        problem.equation,
        ufunc,
        vars2,
        initial_condition=ic_eq,
        boundary_conditions=bc_eqs,
        domain="whole_line" if plan.domain == "interval" else plan.domain,
    )
    return _as_transform_result(res, plan, wrapped=res, native=False)


__all__ = [
    "TransformMethodPlan",
    "build_transform_method_plan",
    "execute_transform_plan",
    "initial_condition_equation",
    "boundary_condition_equations",
]
