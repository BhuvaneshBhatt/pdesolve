from __future__ import annotations

import sympy as sp

from .dispatcher_support import append_trace_step, coerce_result
from .errors import PDEMethodNotApplicable, PDESolveError, PDETransformationError
from .method_names import normalize_method_name
from .planners.coordinator import plan_canonical_problem
from .results import PDESolverTraceStep
from .solver_execution import is_registered_method
from .solvers.coordinator import execute_planned_solver


def _coerce_record(
    raw,
    *,
    attempted,
    trace,
    canonical_eq,
    assumptions,
    dependent,
    variables,
    ics,
    bcs,
    normalize_result,
    max_ops,
):
    return coerce_result(
        raw,
        attempted_methods=attempted,
        trace_steps=trace,
        canonical_eq=canonical_eq,
        assumptions=assumptions,
        dep_expr_or_func=dependent,
        indep_vars=variables,
        ics=ics,
        bcs=bcs,
        normalize_result=normalize_result,
        normalization_max_ops=max_ops,
    )


def _ordered_plan_methods(plan, solver_kwargs):
    methods = []
    seen = set()
    for step in getattr(plan, "steps", ()):
        method = normalize_method_name(step.method)
        if not is_registered_method(method) or method in seen:
            continue
        seen.add(method)
        methods.append(method)

    promoted = {"generalized_clairaut_complete_integral", "quasilinear_implicit"}
    if solver_kwargs.get("prefer_transform"):
        promoted.add("structured_transform")
    if solver_kwargs.get("prefer_separation"):
        promoted.add("separation_framework")
    methods = [m for m in methods if m in promoted] + [m for m in methods if m not in promoted]

    if solver_kwargs.get("prefer_symmetry"):
        sym_methods = {"symmetry_reduction", "post_reduction_auto", "first_order"}
        methods = [m for m in methods if m in sym_methods] + [
            m for m in methods if m not in sym_methods
        ]
    return methods


def _finalize_record(
    record,
    *,
    attempted,
    trace,
    canonical_eq,
    assumptions,
    dependent,
    variables,
    ics,
    bcs,
    normalize_result,
    max_ops,
):
    append_trace_step(
        trace,
        record.method,
        "verify",
        record.verification.get("verified") is not False,
        record.verification.get("message", ""),
        verification=record.verification,
    )
    return _coerce_record(
        record,
        attempted=attempted,
        trace=trace,
        canonical_eq=canonical_eq,
        assumptions=assumptions,
        dependent=dependent,
        variables=variables,
        ics=ics,
        bcs=bcs,
        normalize_result=normalize_result,
        max_ops=max_ops,
    )


def solve_system_problem(
    problem,
    *,
    dependent,
    variables,
    ics,
    assumptions,
    normalize_result,
    max_ops,
    solver_kwargs,
):
    method = "hyperbolic_system"
    attempted = [method]
    trace: list[PDESolverTraceStep] = []
    append_trace_step(trace, method, "attempt", True, "system solver selected")
    raw = execute_planned_solver(problem, method, **solver_kwargs)
    record = _coerce_record(
        raw,
        attempted=attempted,
        trace=trace,
        canonical_eq=sp.Eq(0, 0),
        assumptions=assumptions,
        dependent=dependent,
        variables=variables,
        ics=ics,
        bcs=None,
        normalize_result=normalize_result,
        max_ops=max_ops,
    )
    return _finalize_record(
        record,
        attempted=attempted,
        trace=trace,
        canonical_eq=sp.Eq(0, 0),
        assumptions=assumptions,
        dependent=dependent,
        variables=variables,
        ics=ics,
        bcs=None,
        normalize_result=normalize_result,
        max_ops=max_ops,
    )


def solve_scalar_problem(
    problem,
    *,
    dependent,
    variables,
    ics,
    bcs,
    method,
    assumptions,
    normalize_result,
    max_ops,
    solver_kwargs,
):
    trace: list[PDESolverTraceStep] = []
    attempted: list[str] = []

    if method != "auto":
        attempted.append(method)
        append_trace_step(trace, method, "attempt", True, "explicit method requested")
        raw = execute_planned_solver(problem, method, **solver_kwargs)
        record = _coerce_record(
            raw,
            attempted=attempted,
            trace=trace,
            canonical_eq=problem.equation,
            assumptions=assumptions,
            dependent=dependent,
            variables=variables,
            ics=ics,
            bcs=bcs,
            normalize_result=normalize_result,
            max_ops=max_ops,
        )
        return _finalize_record(
            record,
            attempted=attempted,
            trace=trace,
            canonical_eq=problem.equation,
            assumptions=assumptions,
            dependent=dependent,
            variables=variables,
            ics=ics,
            bcs=bcs,
            normalize_result=normalize_result,
            max_ops=max_ops,
        )

    plan = plan_canonical_problem(
        problem,
        prefer_transform=solver_kwargs.get("prefer_transform", False),
        prefer_separation=solver_kwargs.get("prefer_separation", False),
        prefer_symmetry=solver_kwargs.get("prefer_symmetry", False),
    )
    methods = _ordered_plan_methods(plan, solver_kwargs)
    failures = []
    for candidate in methods:
        attempted.append(candidate)
        append_trace_step(trace, candidate, "attempt", True, "planned method")
        try:
            raw = execute_planned_solver(problem, candidate, **solver_kwargs)
            record = _coerce_record(
                raw,
                attempted=attempted,
                trace=trace,
                canonical_eq=problem.equation,
                assumptions=assumptions,
                dependent=dependent,
                variables=variables,
                ics=ics,
                bcs=bcs,
                normalize_result=normalize_result,
                max_ops=max_ops,
            )
            append_trace_step(trace, record.method, "solve", True, "method succeeded")
            return _finalize_record(
                record,
                attempted=attempted,
                trace=trace,
                canonical_eq=problem.equation,
                assumptions=assumptions,
                dependent=dependent,
                variables=variables,
                ics=ics,
                bcs=bcs,
                normalize_result=normalize_result,
                max_ops=max_ops,
            )
        except (PDEMethodNotApplicable, PDETransformationError) as exc:
            failures.append((candidate, str(exc)))
            append_trace_step(trace, candidate, "solve", False, str(exc), exception=exc)
    raise PDESolveError(f"Could not solve PDE. Tried {methods}. Failures={failures}")


__all__ = ["solve_scalar_problem", "solve_system_problem"]
