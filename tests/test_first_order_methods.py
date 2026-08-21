import sympy as sp
from pdesolve import (
    adapted_coordinate_reduction,
    characteristic_first_integral,
    solve_first_order_linear_pde,
)


def test_characteristic_first_integral_constant_field():
    x, y = sp.symbols("x y")
    inv = characteristic_first_integral(sp.Integer(1), sp.Integer(1), x, y)
    assert sp.simplify(inv - (y - x)) == 0 or sp.simplify(inv - (x - y)) == 0


def test_adapted_coordinate_reduction_smoke():
    x, y = sp.symbols("x y")
    red = adapted_coordinate_reduction(sp.Integer(1), sp.Integer(1), x, y)
    assert red is not None
    assert red.invariant is not None


def test_solve_first_order_linear_pde_smoke():
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) + sp.diff(u(x, y), y), 0)
    res = solve_first_order_linear_pde(eq, u, (x, y))
    assert res.method_family == "first_integral_adapted_coordinates"
