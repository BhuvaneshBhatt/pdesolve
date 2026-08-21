from __future__ import annotations

import sympy as sp

from pdesolve import pdesolve, build_benchmark_suite, run_benchmark_case
from pdesolve.first_order_framework import canonicalize_first_order_nonlinear_pde


def test_first_order_auto_uses_canonical_generalized_clairaut_route():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(
        u(x, y),
        x * sp.diff(u(x, y), x)
        + y * sp.diff(u(x, y), y)
        + sp.sin(sp.diff(u(x, y), x) + sp.diff(u(x, y), y)),
    )
    can = canonicalize_first_order_nonlinear_pde(eq, u(x, y), (x, y))
    assert can.recognized_family == "generalized_clairaut"
    res = pdesolve(eq, u(x, y), (x, y))
    assert (
        "clairaut" in str(getattr(res, "method", "")).lower()
        or "complete_integral" in str(getattr(res, "method", "")).lower()
    )


def test_first_order_auto_uses_canonical_quasilinear_route_with_structured_conditions():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eqs = [
        sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0),
        sp.Eq(u(x, 0), 1 / (1 + x)),
    ]
    res = pdesolve(eqs, u(x, t), (x, t), method="first_order_nonlinear_auto")
    assert res is not None
    assert any(
        tok in str(getattr(res, "method", "")).lower()
        for tok in ("quasilinear", "implicit", "first_order")
    )


def test_benchmark_suite_has_exact_and_stress_cases():
    suite = build_benchmark_suite()
    assert suite.total_cases >= 14
    assert suite.stress_cases
    assert any(
        case.expected_solution is not None
        or case.solution_fragments
        or case.exact_output_kind
        for case in suite.all_cases()
    )


def test_selected_benchmark_cases_run_successfully():
    suite = build_benchmark_suite()
    cases = {case.name: case for case in suite.all_cases()}
    for name in (
        "whole_line_heat",
        "interval_heat_dirichlet",
        "clairaut",
        "linear_flux_profile",
    ):
        outcome = run_benchmark_case(cases[name])
        assert outcome.success, (name, outcome.message)


def test_benchmark_case_exact_and_method_checks():
    suite = build_benchmark_suite()
    cases = {case.name: case for case in suite.all_cases()}
    outcome = run_benchmark_case(cases["linear_flux_profile"])
    assert outcome.success, outcome.message
    assert outcome.contains_verified is True
    assert outcome.method_hint_verified is True


def test_benchmark_suite_includes_large_symbolic_stress_cases():
    suite = build_benchmark_suite()
    stress_names = {case.name for case in suite.stress_cases}
    assert "whole_line_heat_polynomial_stress" in stress_names
    assert "wave_trigonometric_stress" in stress_names
    assert "rectangle_laplace_polynomial_boundary_stress" in stress_names
    assert "nonlinear_complete_integral_stress" in stress_names


def test_benchmark_suite_has_more_exact_solution_assertions():
    suite = build_benchmark_suite()
    exact_names = {
        case.name for case in suite.all_cases() if case.expected_solution is not None
    }
    assert {
        "transport_ivp_gaussian",
        "clairaut",
        "linear_flux_profile",
        "whole_line_heat",
        "interval_wave_dirichlet",
        "whole_line_dalembert",
        "hyperbolic_system",
    } <= exact_names


def test_selected_exact_benchmark_cases_verify_symbolically():
    suite = build_benchmark_suite()
    cases = {case.name: case for case in suite.all_cases()}
    for name in (
        "transport_ivp_gaussian",
        "clairaut",
        "linear_flux_profile",
        "whole_line_heat",
        "interval_wave_dirichlet",
        "whole_line_dalembert",
        "hyperbolic_system",
    ):
        outcome = run_benchmark_case(cases[name])
        assert outcome.success, (name, outcome.message)
        assert outcome.exact_verified is True
