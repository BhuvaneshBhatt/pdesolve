from collections.abc import Mapping

import sympy as sp

from pdesolve import pdesolve


def _method_family_of(res):
    mf = getattr(res, "method_family", None)
    if mf:
        return mf
    details = getattr(res, "details", None)
    if isinstance(details, Mapping):
        return details.get("method_family")
    return None


def test_pipeline_uses_unified_transform_for_heat_with_initial_data():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    ic = sp.Eq(u(x, 0), sp.exp(-(x**2)))
    res = pdesolve(eq, u, (x, t), ics=ic, method="unified_transform", domain="whole_line")
    assert _method_family_of(res) == "unified_transform_whole_line"


def test_pipeline_integrates_first_order_or_constant_coefficient_solver():
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) + sp.diff(u(x, y), y), 0)
    res = pdesolve(eq, u, (x, y))
    assert _method_family_of(res) in {"first_integral_adapted_coordinates", "homogeneous"}


def test_pipeline_uses_constant_coefficient_solver():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x), sp.exp(x + t))
    res = pdesolve(eq, u, (x, t))
    method = getattr(res, "method", "") or ""
    details = getattr(res, "details", {}) or {}
    assert (
        "constant_coefficient" in method
        or "exponential" in method
        or "polynomial" in method
        or details.get("solver_family") == "constant_coefficient"
    )
