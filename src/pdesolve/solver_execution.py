from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.polys.polyerrors import PolynomialError

from .complete_integral_helpers import (
    solve_charpit_complete_integral_2vars,
    solve_complete_integral_pde,
    solve_generalized_clairaut_complete_integral,
    solve_jacobi_complete_integral,
)
from .conditions import extract_equations_by_role, summarize_condition_model
from .constant_coeff import pdesolve_constant_coefficient
from .errors import PDEMethodNotApplicable
from .first_order_framework import canonicalize_first_order_nonlinear_pde, execute_first_order_plan
from .first_order_nonlinear import solve_via_invariant_reduction
from .hyperbolic_system import solve_hyperbolic_system
from .kernels import execute_kernel_plan
from .method_names import normalize_method_name
from .results import (
    BasePDEResult,
    ClosedFormPDEResult,
    EigenfunctionExpansionResult,
    FundamentalSolutionResult,
    GreenFunctionResult,
    ImplicitPDEResult,
    KernelRepresentationResult,
    SeriesPDEResult,
    SolverMethodResult,
    SystemPDEResult,
    TransformPDEResult,
    UnsolvedButReducedResult,
    WeakSolutionResult,
)
from .separation_framework import execute_separation_plan
from .transform_framework import execute_transform_plan
from .unified_transform import solve_unified_transform


@dataclass(frozen=True)
class SolverExecutionContext:
    problem: Any
    method: str
    classical_mod: Any
    options: dict[str, Any] = field(default_factory=dict)


def _ctx_adapter(handler):
    def _wrapped(ctx: SolverExecutionContext):
        return handler(ctx.problem, ctx.method, ctx.classical_mod, **dict(ctx.options))

    return _wrapped


def _classical():
    from . import classical_methods as classical_mod

    return classical_mod


def _problem_condition_model(problem):
    if getattr(problem, "canonical_representation", None) is not None:
        details = getattr(problem.canonical_representation, "details", {}) or {}
        cm = details.get("condition_model")
        if cm is not None:
            return cm
    return problem.details.get("condition_model") if getattr(problem, "details", None) else None


def _extract_condition_payloads(problem):
    cm = _problem_condition_model(problem)
    summary = summarize_condition_model(cm) if cm is not None else {}
    ics_payload = dict(problem.ics) if isinstance(problem.ics, dict) else {}
    bcs_payload = dict(problem.bcs) if isinstance(problem.bcs, dict) else {}
    if cm is not None:
        init_eqs = extract_equations_by_role(cm, "initial")
        bc_eqs = extract_equations_by_role(cm, "boundary")
        if "equations" not in ics_payload and init_eqs:
            ics_payload = {**ics_payload, "equations": init_eqs}
        if "equations" not in bcs_payload and bc_eqs:
            bcs_payload = {**bcs_payload, "equations": bc_eqs}
        if init_eqs and "equation" not in ics_payload:
            ics_payload = {**ics_payload, "equation": init_eqs[0], "initial_equation": init_eqs[0]}
        ts = summary.get("time_slices", ())
        if ts and "curve_value" not in ics_payload:
            ics_payload["curve_value"] = ts[0]
        for cond, ck in zip(
            getattr(cm, "initial_conditions", ()), summary.get("initial_kinds", ()), strict=True
        ):
            rhs = (
                cond.equation.rhs
                if isinstance(getattr(cond, "equation", None), sp.Equality)
                else None
            )
            if ck == "profile" and rhs is not None and "initial_profile" not in ics_payload:
                ics_payload["initial_profile"] = rhs
                ics_payload.setdefault("initial_displacement", rhs)
            elif ck == "velocity" and rhs is not None and "initial_velocity" not in ics_payload:
                ics_payload["initial_velocity"] = rhs
        geom = problem.details.get("domain_geometry") if getattr(problem, "details", None) else None
        if geom is not None:
            if "type" not in bcs_payload:
                if geom.kind == "interval" and set(summary.get("boundary_kinds", ())) == {
                    "dirichlet"
                }:
                    bcs_payload["type"] = "dirichlet_homogeneous_interval"
                elif geom.kind == "interval" and set(summary.get("boundary_kinds", ())) == {
                    "neumann"
                }:
                    bcs_payload["type"] = "neumann_homogeneous_interval"
            for key, val in getattr(geom, "extents", {}).items():
                if key == "x" and isinstance(val, tuple) and len(val) == 2:
                    bcs_payload.setdefault("length", val[1] - val[0])
                else:
                    bcs_payload.setdefault(f"{key}_extent", val)
    return cm, summary, ics_payload, bcs_payload


def _metadata(problem, extra=None):
    md = {
        "canonical_representation": getattr(problem, "canonical_representation", None),
        "condition_model": getattr(problem, "details", {}).get("condition_model"),
        "domain_geometry": getattr(problem, "details", {}).get("domain_geometry"),
        "boundary_model": getattr(problem, "details", {}).get("boundary_model"),
        "condition_analysis": getattr(problem, "details", {}).get("condition_analysis"),
        "problem": problem,
    }
    if extra:
        md.update(extra)
    return md


def _standardize_result(raw: Any, problem, default_method: str) -> BasePDEResult:
    if isinstance(raw, BasePDEResult):
        md = dict(getattr(raw, "metadata", {}) or {})
        md.update(_metadata(problem))
        cls = type(raw)
        kwargs = dict(
            method=raw.method,
            solution=raw.solution,
            classification=getattr(raw, "classification", None),
            assumptions=getattr(raw, "assumptions", True),
            verification=dict(getattr(raw, "verification", {}) or {}),
            metadata=md,
            reduced_problem=getattr(raw, "reduced_problem", None),
            warnings=tuple(getattr(raw, "warnings", ()) or ()),
        )
        if isinstance(raw, EigenfunctionExpansionResult):
            kwargs["series_terms"] = getattr(raw, "series_terms", None)
            kwargs["eigen_data"] = getattr(raw, "eigen_data", None)
            return EigenfunctionExpansionResult(**kwargs)
        if isinstance(raw, SeriesPDEResult) and not isinstance(raw, EigenfunctionExpansionResult):
            kwargs["series_terms"] = getattr(raw, "series_terms", None)
            return SeriesPDEResult(**kwargs)
        if isinstance(raw, TransformPDEResult):
            kwargs["transform_data"] = getattr(raw, "transform_data", None)
            return TransformPDEResult(**kwargs)
        if isinstance(raw, WeakSolutionResult):
            kwargs["admissibility"] = dict(getattr(raw, "admissibility", {}) or {})
            return WeakSolutionResult(**kwargs)
        if isinstance(raw, SystemPDEResult):
            kwargs["system_size"] = getattr(raw, "system_size", 1)
            kwargs["transform"] = getattr(raw, "transform", None)
            kwargs["characteristic_variables"] = getattr(raw, "characteristic_variables", None)
            return SystemPDEResult(**kwargs)
        if isinstance(raw, FundamentalSolutionResult):
            kwargs["kernel"] = getattr(raw, "kernel", getattr(raw, "solution", None))
            kwargs["source_point"] = getattr(raw, "source_point", None)
            kwargs["operator_family"] = getattr(raw, "operator_family", None)
            return FundamentalSolutionResult(**kwargs)
        if isinstance(raw, GreenFunctionResult):
            kwargs["kernel"] = getattr(raw, "kernel", getattr(raw, "solution", None))
            kwargs["source_point"] = getattr(raw, "source_point", None)
            kwargs["operator_family"] = getattr(raw, "operator_family", None)
            kwargs["boundary_type"] = getattr(raw, "boundary_type", None)
            return GreenFunctionResult(**kwargs)
        if isinstance(raw, KernelRepresentationResult):
            kwargs["kernel"] = getattr(raw, "kernel", getattr(raw, "solution", None))
            kwargs["source_point"] = getattr(raw, "source_point", None)
            kwargs["operator_family"] = getattr(raw, "operator_family", None)
            return KernelRepresentationResult(**kwargs)
        return cls(**kwargs)
    method = getattr(raw, "method", default_method)
    solution = getattr(raw, "solution", raw)
    details = dict(getattr(raw, "details", {}) or getattr(raw, "metadata", {}) or {})
    method_family = getattr(raw, "method_family", None)
    if method_family is not None:
        details.setdefault("method_family", method_family)
    details.update(_metadata(problem, {"raw_result": raw}))
    cls = ClosedFormPDEResult
    if isinstance(solution, dict) or "system" in method:
        cls = SystemPDEResult
    elif (
        "shock" in method
        or "rarefaction" in method
        or ("conservation" in method and ("admissibility" in details or "weak" in method))
    ):
        cls = WeakSolutionResult
    elif (
        any(tok in method for tok in ("transform", "fourier", "laplace"))
        or details.get("transform_data") is not None
    ):
        cls = TransformPDEResult
    elif (
        "series" in method
        or isinstance(solution, sp.Sum)
        or details.get("separation_plan") is not None
    ):
        cls = (
            EigenfunctionExpansionResult
            if ("eigen" in method or "basis" in str(details.get("plan", "")) or "series" in method)
            else SeriesPDEResult
        )
    elif "implicit" in method or isinstance(solution, (tuple, list)):
        cls = ImplicitPDEResult
    elif (
        method in {"kernel_fundamental_solution", "kernel_green_function"}
        or details.get("kernel_plan") is not None
    ):
        cls = (
            FundamentalSolutionResult
            if method == "kernel_fundamental_solution"
            else GreenFunctionResult
        )
    elif method in {"classification_only", "reduced_unsolved"}:
        cls = UnsolvedButReducedResult
    common = dict(
        method=method,
        solution=solution,
        classification=details.get(
            "classification",
            getattr(problem.profile, "canonical_family", None)
            if getattr(problem, "profile", None) is not None
            else None,
        ),
        assumptions=getattr(problem, "assumptions", True),
        verification={},
        metadata=details,
        reduced_problem=details.get("reduced_problem"),
        warnings=tuple(details.get("warnings", ()) or ()),
    )
    if cls is TransformPDEResult:
        return TransformPDEResult(**common, transform_data=details.get("transform_data"))
    if cls is WeakSolutionResult:
        return WeakSolutionResult(
            **common, admissibility=dict(details.get("admissibility", {}) or {})
        )
    if cls is EigenfunctionExpansionResult:
        return EigenfunctionExpansionResult(
            **common, series_terms=details.get("terms"), eigen_data=details.get("eigen_data")
        )
    if cls is SeriesPDEResult:
        return SeriesPDEResult(**common, series_terms=details.get("terms"))
    if cls is SystemPDEResult:
        return SystemPDEResult(
            **common,
            system_size=len(solution)
            if isinstance(solution, dict)
            else details.get("system_size", 1),
            transform=details.get("transform"),
            characteristic_variables=details.get("characteristic_variables"),
        )
    if cls is FundamentalSolutionResult:
        return FundamentalSolutionResult(
            **common,
            kernel=details.get("kernel", solution),
            source_point=details.get("source_point"),
            operator_family=details.get("operator_family"),
        )
    if cls is GreenFunctionResult:
        return GreenFunctionResult(
            **common,
            kernel=details.get("kernel", solution),
            source_point=details.get("source_point"),
            operator_family=details.get("operator_family"),
            boundary_type=details.get("boundary_type"),
        )
    return cls(**common)


def _first_order_nonlinear(problem, method, classical_mod, **kwargs):
    eq = problem.equation
    uexpr = problem.dep_function
    vars_ = problem.indep_vars
    can1 = (getattr(problem.canonical_representation, "details", {}) or {}).get(
        "first_order_canonical"
    )
    if can1 is None and len(vars_) >= 2:
        can1 = canonicalize_first_order_nonlinear_pde(eq, uexpr, vars_)
    if can1 is None:
        raise NotImplementedError("No canonical first-order representation available.")
    return execute_first_order_plan(problem, classical_mod, canonical=can1, **kwargs)


def _invariant_reduction(problem, method, classical_mod, **kwargs):
    return solve_via_invariant_reduction(
        problem.equation,
        problem.dep_function,
        problem.indep_vars,
        ics=problem.ics,
        bcs=problem.bcs,
        assumptions=problem.assumptions,
        **kwargs,
    )


def _classification_only(problem, method, classical_mod, **kwargs):
    return UnsolvedButReducedResult(
        method="classification_only",
        solution=problem.equation,
        classification=getattr(problem.profile, "canonical_family", None)
        if getattr(problem, "profile", None) is not None
        else None,
        assumptions=problem.assumptions,
        metadata=_metadata(problem),
        reduced_problem=problem.equation,
    )


def _first_order(problem, method, classical_mod, **kwargs):
    if problem.profile.first_order_linear is not None:
        return classical_mod.solve_first_order_linear_pde_pdsolve(
            problem.equation, problem.dep_function, problem.indep_vars
        )
    return classical_mod.solve_first_order_pde_characteristic(
        problem.equation, problem.dep_function, problem.indep_vars
    )


def _transport(problem, method, classical_mod, **kwargs):
    _, _, ics, _ = _extract_condition_payloads(problem)
    return classical_mod.solve_transport_ivp(
        problem.equation,
        problem.dep_function,
        problem.indep_vars,
        initial_profile=ics.get("initial_profile"),
        initial_curve_value=ics.get("curve_value", 0),
    )


def _quasilinear(problem, method, classical_mod, **kwargs):
    _, _, ics, _ = _extract_condition_payloads(problem)
    return classical_mod.solve_quasilinear_pde_characteristics_implicit(
        problem.equation,
        problem.dep_function,
        problem.indep_vars,
        initial_profile=ics.get("initial_profile"),
        initial_curve_value=ics.get("curve_value", 0),
    )


def _conservation(problem, method, classical_mod, **kwargs):
    recog = classical_mod.detect_scalar_conservation_law_family(
        problem.equation, problem.dep_function, problem.indep_vars
    )
    return WeakSolutionResult(
        method="conservation_law_analysis",
        solution=recog.normalized_equation,
        classification="scalar_conservation_law",
        assumptions=problem.assumptions,
        metadata=_metadata(problem, {"family": recog}),
        admissibility={},
    )


def _burgers(problem, method, classical_mod, **kwargs):
    _, _, ics, _ = _extract_condition_payloads(problem)
    x, t = problem.indep_vars[:2]
    return classical_mod.solve_burgers_ivp_characteristic_formal(
        problem.dep_function, x=x, t=t, initial_profile=ics.get("initial_profile")
    )


def _wave(problem, method, classical_mod, **kwargs):
    _, _, ics, _ = _extract_condition_payloads(problem)
    x, t = problem.indep_vars[:2]
    return classical_mod.solve_wave_equation_1d_ivp(
        problem.dep_function,
        x=x,
        t=t,
        wave_speed=kwargs.get("wave_speed", 1),
        initial_displacement=ics.get("initial_displacement"),
        initial_velocity=ics.get("initial_velocity"),
    )


def _heat(problem, method, classical_mod, **kwargs):
    _, _, ics, _ = _extract_condition_payloads(problem)
    x, t = problem.indep_vars[:2]
    return classical_mod.solve_heat_equation_1d_whole_line_ivp(
        problem.dep_function,
        x=x,
        t=t,
        diffusivity=kwargs.get("diffusivity", 1),
        initial_profile=ics.get("initial_profile"),
    )


def _fourier_heat(problem, method, classical_mod, **kwargs):
    _, _, ics, _ = _extract_condition_payloads(problem)
    x, t = problem.indep_vars[:2]
    return classical_mod.solve_heat_equation_1d_fourier_transform(
        problem.dep_function,
        x=x,
        t=t,
        diffusivity=kwargs.get("diffusivity", 1),
        initial_profile=ics.get("initial_profile"),
    )


def _series(problem, method, classical_mod, **kwargs):
    if method == "separation_of_variables":
        raw = classical_mod.separate_variables_structured(
            problem.equation,
            problem.dep_function,
            problem.indep_vars,
            assumptions=problem.assumptions,
            bcs=problem.bcs,
        )
        return SolverMethodResult(
            method_family="separation_of_variables",
            solution=raw.ansatz,
            details={
                "family": raw.family,
                "separated_odes": raw.separated_odes,
                "separation_constants": raw.separation_constants,
                "basis_hint": raw.basis_hint,
                "raw_result": raw,
            },
        )
    return execute_separation_plan(problem, classical_mod, **kwargs)


def _symmetry(problem, method, classical_mod, **kwargs):
    _, _, ics, _ = _extract_condition_payloads(problem)
    dep_func = getattr(problem.dep_function, "func", None)
    wrapped_dep = [
        atom
        for atom in problem.equation.atoms(AppliedUndef)
        if atom.func != dep_func and atom.has(problem.dep_function)
    ]
    if wrapped_dep:
        raise PDEMethodNotApplicable(
            "Point-symmetry reduction does not model custom operators applied to the dependent variable."
        )
    if (
        kwargs.get("prefer_symmetry")
        and problem.profile.order == 1
        and problem.profile.first_order_linear is not None
        and ics.get("initial_profile") is None
    ):
        return classical_mod.solve_first_order_linear_pde_pdsolve(
            problem.equation, problem.dep_function, problem.indep_vars
        )
    try:
        return classical_mod._solve_via_symmetry_workflow(
            problem.equation,
            problem.dep_function,
            problem.indep_vars,
            assumptions=problem.assumptions,
            max_symmetry_steps=kwargs.get("max_symmetry_steps", 2),
        )
    except PolynomialError as exc:
        raise PDEMethodNotApplicable(
            "Polynomial symmetry ansatz is not applicable to this coefficient structure."
        ) from exc


def _post_reduction(problem, method, classical_mod, **kwargs):
    return classical_mod.solve_reduced_equation_auto(
        problem.equation,
        ics=problem.ics,
        bcs=problem.bcs,
        assumptions=problem.assumptions,
        max_symmetry_steps=kwargs.get("max_symmetry_steps", 2),
    )


def _complete_integral(problem, method, classical_mod, **kwargs):
    if method == "generalized_clairaut_complete_integral":
        return solve_generalized_clairaut_complete_integral(
            problem.equation, problem.dep_function, problem.indep_vars, **kwargs
        )
    if method == "charpit":
        return solve_charpit_complete_integral_2vars(
            problem.equation, problem.dep_function, problem.indep_vars
        )
    if method == "complete_integral":
        return solve_complete_integral_pde(
            problem.equation,
            problem.dep_function,
            problem.indep_vars,
            assumptions=problem.assumptions,
            **kwargs,
        )
    return solve_jacobi_complete_integral(
        problem.equation, problem.dep_function, problem.indep_vars, **kwargs
    )


def _transform(problem, method, classical_mod, **kwargs):
    if method == "unified_transform":
        _, _, ics, bcs = _extract_condition_payloads(problem)
        ic_eq = ics.get("equation")
        if ic_eq is None:
            ic_eq = ics.get("initial_equation")
        bc_eqs = list(bcs.get("equations", ()) or ())
        return solve_unified_transform(
            problem.equation,
            problem.dep_function.func,
            problem.indep_vars[:2],
            initial_condition=ic_eq,
            boundary_conditions=bc_eqs,
            domain=kwargs.get("domain", "whole_line"),
        )
    return execute_transform_plan(problem, classical_mod=classical_mod, **kwargs)


def _system(problem, method, classical_mod, **kwargs):
    eqs = problem.details.get("raw_equations")
    unknowns = problem.details.get("raw_unknowns")
    ic_eqs = problem.details.get("raw_ics") or []
    return solve_hyperbolic_system(
        eqs, ic_eqs, unknowns, vars=(problem.indep_vars[0], problem.indep_vars[1])
    )


def _constant_coeff(problem, method, classical_mod, **kwargs):
    _, _, ics, bcs = _extract_condition_payloads(problem)
    return pdesolve_constant_coefficient(
        problem.equation,
        problem.dep_function,
        problem.indep_vars,
        ics=ics,
        bcs=bcs,
        assumptions=problem.assumptions,
    )


def _kernel(problem, method, classical_mod, **kwargs):
    return execute_kernel_plan(problem, **kwargs)


_RAW_METHOD_REGISTRY: dict[str, Callable[..., Any]] = {
    "first_order_nonlinear_auto": _first_order_nonlinear,
    "invariant_reduction_auto": _invariant_reduction,
    "classification_only": _classification_only,
    "first_order": _first_order,
    "transport_ivp": _transport,
    "quasilinear_implicit": _quasilinear,
    "conservation_law": _conservation,
    "burgers_implicit": _burgers,
    "wave_dalembert": _wave,
    "heat_whole_line": _heat,
    "fourier_heat": _fourier_heat,
    "heat_dirichlet_series": _series,
    "heat_neumann_series": _series,
    "heat_robin_series": _series,
    "wave_dirichlet_series": _series,
    "separation_framework": _series,
    "separation_of_variables": _series,
    "laplace_rectangle_dirichlet_series": _series,
    "heat_half_line_transform": _transform,
    "heat_laplace_transform": _transform,
    "laplace_fourier_heat": _transform,
    "wave_laplace_transform": _transform,
    "wave_laplace_sine_transform": _transform,
    "structured_transform": _transform,
    "unified_transform": _transform,
    "symmetry_reduction": _symmetry,
    "post_reduction_auto": _post_reduction,
    "generalized_clairaut_complete_integral": _complete_integral,
    "charpit": _complete_integral,
    "complete_integral": _complete_integral,
    "jacobi": _complete_integral,
    "hyperbolic_system": _system,
    "constant_coefficient_inverse_operator": _constant_coeff,
    "kernel_fundamental_solution": _kernel,
    "kernel_green_function": _kernel,
}


_METHOD_REGISTRY: dict[str, Callable[[SolverExecutionContext], Any]] = {
    name: _ctx_adapter(handler) for name, handler in _RAW_METHOD_REGISTRY.items()
}


@contextmanager
def temporary_method_handler(method: str, handler):
    """Temporarily replace one solver handler for dispatch-contract tests.

    The registry is restored even when the wrapped call raises, so tests do not
    depend on pytest-specific mutation helpers or leak state across cases.
    """
    key = normalize_method_name(method)
    previous = _METHOD_REGISTRY.get(key)
    _METHOD_REGISTRY[key] = handler
    try:
        yield
    finally:
        if previous is None:
            _METHOD_REGISTRY.pop(key, None)
        else:
            _METHOD_REGISTRY[key] = previous


def registered_method_names() -> tuple[str, ...]:
    """Return canonical method keys accepted by the execution registry."""
    return tuple(sorted(_METHOD_REGISTRY))


def is_registered_method(method: str) -> bool:
    return normalize_method_name(method) in _METHOD_REGISTRY


def solve_with_canonical_problem(problem, method: str, **kwargs) -> BasePDEResult:
    classical_mod = _classical()
    method = normalize_method_name(method)
    handler = _METHOD_REGISTRY.get(method)
    if handler is None:
        raise PDEMethodNotApplicable(f"Unsupported method: {method}")
    ctx = SolverExecutionContext(
        problem=problem, method=method, classical_mod=classical_mod, options=dict(kwargs)
    )
    raw = handler(ctx)
    return _standardize_result(raw, problem, method)


__all__ = [
    "SolverExecutionContext",
    "registered_method_names",
    "is_registered_method",
    "solve_with_canonical_problem",
]
