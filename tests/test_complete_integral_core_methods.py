import sympy as sp

import pdesolve as rle


def test_charpit_autonomous_complete_integral_eikonal_like():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) ** 2 + sp.diff(u(x, y), y) ** 2, 1)
    res = rle.solve_charpit_complete_integral_2vars(eq, u(x, y), (x, y))
    assert res.solutions
    assert any(sol.lhs == u(x, y) for sol in res.solutions)


def test_jacobi_autonomous_complete_integral_3vars():
    x, y, z = sp.symbols("x y z", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y, z), x) ** 2, sp.diff(u(x, y, z), y) + sp.diff(u(x, y, z), z))
    res = rle.solve_jacobi_complete_integral(eq, u(x, y, z), (x, y, z))
    assert res.solutions


def test_linear_constant_coefficient_bvp_single_family():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0)
    res = rle.solve_linear_constant_coefficient_pde_bvp_2d(
        eq, u(x, t), (x, t), ics={"initial_profile": x**2, "curve_value": 0}
    )
    assert isinstance(res.solution, sp.Equality)


def test_free_free_beam_solver():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")
    res = rle.solve_euler_bernoulli_beam_freefree_ibvp(
        u(x, t),
        x=x,
        t=t,
        length=1,
        stiffness=1,
        initial_displacement=sp.cos(sp.pi * x),
        initial_velocity=0,
        n_terms=3,
    )
    assert isinstance(res.solution, sp.Equality)
