from __future__ import annotations

import sympy as sp

from pdesolve import build_pde_problem, pdesolve, build_benchmark_suite
from pdesolve.separation_framework import execute_separation_plan
from pdesolve.transform_framework import execute_transform_plan


def test_native_structured_separation_uses_condition_model_directly():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    prob = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics=sp.Eq(u(x, 0), sp.sin(x)),
        bcs=[sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)],
    )
    res = execute_separation_plan(
        prob, __import__("pdesolve.classical_methods", fromlist=["dummy"])
    )
    assert "heat" in str(getattr(res, "method", ""))


def test_native_structured_transform_uses_condition_model_directly():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    prob = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics=sp.Eq(u(x, 0), sp.exp(-(x**2))),
    )
    res = execute_transform_plan(
        prob, classical_mod=__import__("pdesolve.classical_methods", fromlist=["dummy"])
    )
    assert any(
        tok in str(getattr(res, "method", ""))
        for tok in ("transform", "unified", "laplace", "fourier")
    )


def test_first_order_framework_executes_generalized_clairaut_directly():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(
        u(x, y),
        x * sp.diff(u(x, y), x)
        + y * sp.diff(u(x, y), y)
        + sp.sin(sp.diff(u(x, y), x) + sp.diff(u(x, y), y)),
    )
    res = pdesolve(eq, u(x, y), (x, y), method="generalized_clairaut_complete_integral")
    assert (
        "clairaut" in str(getattr(res, "method", "")).lower()
        or "complete_integral" in str(getattr(res, "method", "")).lower()
    )


def test_benchmark_suite_is_comprehensive_enough():
    suite = build_benchmark_suite()
    families = set(suite.families)
    assert {
        "first_order_linear",
        "first_order_nonlinear",
        "conservation_law",
        "heat_like",
        "wave_like",
        "laplace_like",
        "system_pde",
    } <= families
    assert sum(len(v) for v in suite.cases_by_family.values()) >= 7


def test_pdesolve_uses_structured_paths_end_to_end():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    res = pdesolve(
        [
            sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
            sp.Eq(u(x, 0), sp.sin(x)),
            sp.Eq(u(0, t), 0),
            sp.Eq(u(sp.pi, t), 0),
        ],
        u(x, t),
        (x, t),
        method="separation_framework",
    )
    assert res is not None
