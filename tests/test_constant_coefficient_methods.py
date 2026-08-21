import sympy as sp

from pdesolve.classical import pdesolve_constant_coefficient
from pdesolve.constant_coeff import (
    _cc_operator_apply_from_terms,
    build_constant_coefficient_homogeneous_solution,
    detect_linear_constant_coefficient_pde,
    invert_constant_coefficient_operator_on_forcing,
)


def _residual(eq, expr, uexpr, vars_):
    ccpde = detect_linear_constant_coefficient_pde(eq, uexpr, vars_)
    return sp.simplify(_cc_operator_apply_from_terms(ccpde.operator_terms, expr, vars_) - ccpde.rhs)


def test_point_condition_fit_reports_linear_algebra_strategy_and_method_family():
    x = sp.symbols("x", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x), x, 2) - u(x), 0)
    res = pdesolve_constant_coefficient(
        eq,
        u(x),
        (x,),
        bcs={"point_value": 0, "point_conditions": {0: 1, 1: 0}},
    )
    assert res.details["fit_strategy"] == "linear_algebra_over_generators"
    assert res.details["method_family"] == "condition_fitting"
    report = res.details["method_family_report"]
    assert report["method_family"] == "condition_fitting"
    assert report["fit_strategy"] == "linear_algebra_over_generators"


def test_1d_characteristic_root_fallback_builds_homogeneous_solution():
    x = sp.symbols("x")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x), x, 2) + sp.diff(u(x), x) + u(x), 0)
    res = build_constant_coefficient_homogeneous_solution(eq, u(x), (x,))
    assert res.method == "constant_coefficient_homogeneous_characteristic_roots"
    assert sp.simplify(_residual(eq, res.solution.rhs, u(x), (x,))) == 0
    assert len(res.details["families"]) == 2


def test_trig_hyperbolic_terms_use_structured_exponential_amplitude_engine():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    rhs = x * sp.cos(t) + sp.sinh(t)
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t), rhs)
    res = invert_constant_coefficient_operator_on_forcing(eq, u(x, t), (x, t))
    assert res.method == "constant_coefficient_trig_hyperbolic"
    assert res.details["engine"] == "exponential_amplitude"
    assert res.details["method_family"] == "trig_hyperbolic_exponential_amplitude"
    assert len(res.details["parts"]) == 4
    assert (
        sp.simplify(
            _residual(eq, res.solution, u(x, t), (x, t))
            .rewrite(sp.sin)
            .rewrite(sp.cos)
            .rewrite(sp.sinh)
            .rewrite(sp.cosh)
        )
        == 0
    )


def test_constant_coefficient_solver_reports_method_families():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x), sp.exp(x + t))
    res = pdesolve_constant_coefficient(eq, u(x, t), (x, t))
    report = res.details["method_family_report"]
    assert report["selected_method"] == res.method
    assert report["particular_method"] is not None
    assert report["particular_method_family"] in {"exponential", "exponential_amplitude"}
    assert report["homogeneous_method_family"] == "homogeneous"
    assert sp.simplify(_residual(eq, res.solution.rhs, u(x, t), (x, t))) == 0
