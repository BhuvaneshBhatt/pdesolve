import sympy as sp

from pdesolve.constant_coeff import pdesolve_constant_coefficient
from pdesolve.first_order_linear import solve_first_order_linear_pde
from pdesolve.first_order_nonlinear import solve_first_order_quasilinear_pde
from pdesolve.hyperbolic_system import solve_hyperbolic_system
from pdesolve.special_pdes import solve_special_pde
from pdesolve.unified_transform import solve_unified_transform


def test_constant_coefficient_method():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x), 0)
    res = pdesolve_constant_coefficient(eq, u, (x, t))
    assert res is not None


def test_first_order_linear_method():
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) + sp.diff(u(x, y), y), 0)
    res = solve_first_order_linear_pde(eq, u, (x, y))
    assert res.invariant is not None


def test_first_order_quasilinear_method():
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    eq = sp.Eq(2 * sp.diff(u(x, y), x) + sp.diff(u(x, y), y), 3)
    res = solve_first_order_quasilinear_pde(eq, u, (x, y))
    assert res.solution is not None


def test_special_pde_method():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0)
    res = solve_special_pde(eq, u, (x, t))
    assert res is not None


def test_unified_transform_whole_line():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    ic = sp.Eq(u(x, 0), sp.exp(-(x**2)))
    eq = sp.Eq(sp.I * sp.diff(u(x, t), t) + sp.diff(u(x, t), x, 2), 0)
    res = solve_unified_transform(eq, u, (x, t), initial_condition=ic, domain="whole_line")
    assert res.domain == "whole_line"


def test_unified_transform_half_line():
    x, t = sp.symbols("x t", positive=True)
    u = sp.Function("u")
    ic = sp.Eq(u(x, 0), sp.exp(-x))
    bc = [sp.Eq(u(0, t), 0)]
    eq = sp.Eq(sp.I * sp.diff(u(x, t), t) + sp.diff(u(x, t), x, 2), 0)
    res = solve_unified_transform(
        eq, u, (x, t), initial_condition=ic, boundary_conditions=bc, domain="half_line"
    )
    assert res.domain == "half_line"


def test_hyperbolic_system_method():
    x, t = sp.symbols("x t")
    u1 = sp.Function("u1")
    u2 = sp.Function("u2")
    eqs = [
        sp.Eq(sp.diff(u1(t, x), t), sp.diff(u2(t, x), x)),
        sp.Eq(sp.diff(u2(t, x), t), sp.diff(u1(t, x), x)),
    ]
    ics = [sp.Eq(u1(0, x), sp.sin(x)), sp.Eq(u2(0, x), sp.cos(x))]
    res = solve_hyperbolic_system(eqs, ics, [u1, u2], (x, t))
    assert len(res.solution_map) == 2
