import sympy as sp

from pdesolve import recognize_evolution_pde, solve_unified_transform


def test_recognize_schrodinger_like():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.I * sp.diff(u(x, t), t) + sp.diff(u(x, t), x, 2), 0)
    profile = recognize_evolution_pde(eq, u, (x, t))
    assert profile is not None
    assert profile.family_name == "schrodinger_like"


def test_half_line_schrodinger_zero_dirichlet_is_not_formal():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.I * sp.diff(u(x, t), t) + sp.diff(u(x, t), x, 2), 0)
    ic = sp.Eq(u(x, 0), sp.sin(x))
    bc = [sp.Eq(u(0, t), 0)]
    res = solve_unified_transform(
        eq, u, (x, t), initial_condition=ic, boundary_conditions=bc, domain="half_line"
    )
    assert res.method_family == "unified_transform_half_line_schrodinger_dirichlet_zero"
    assert res.is_formal is False
