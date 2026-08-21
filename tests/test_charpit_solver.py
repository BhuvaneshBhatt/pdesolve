import sympy as sp
import pdesolve as rle


def test_pfaffian_integrating_factor_depvar_ode_case():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    U = sp.Symbol("U_pf")
    sol = rle.integrate_pfaffian_equation([U, 0], u(x, y), (x, y), dependent_symbol=U)
    assert sol.lhs == u(x, y)
    # u_x = u and u_y = 0
    assert sp.simplify(sp.diff(sol.rhs, x) - sol.rhs) == 0
    assert sp.simplify(sp.diff(sol.rhs, y)) == 0


def test_complete_integral_result_can_verify_implicit_or_explicit_branch():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x), u(x, y))
    res = rle.solve_charpit_complete_integral_2vars(eq, u(x, y), (x, y))
    assert res.solutions
    assert res.verification
    assert any(v["verified"] for v in res.verification)
