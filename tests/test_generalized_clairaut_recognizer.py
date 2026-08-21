import sympy as sp

from pdesolve import pdesolve
from pdesolve.complete_integral_helpers import recognize_generalized_clairaut_pde


def test_recognize_generalized_clairaut_complete_integral():
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    eq = sp.Eq(
        u(x, y),
        x * sp.diff(u(x, y), x)
        + y * sp.diff(u(x, y), y)
        + sp.sin(sp.diff(u(x, y), x) + sp.diff(u(x, y), y)),
    )
    rec = recognize_generalized_clairaut_pde(eq, u(x, y), (x, y))
    assert rec.recognized is True
    assert sp.simplify(rec.phi - sp.sin(rec.gradients[0] + rec.gradients[1])) == 0


def test_pdesolve_auto_short_circuits_generalized_clairaut():
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    eq = sp.Eq(
        u(x, y),
        x * sp.diff(u(x, y), x)
        + y * sp.diff(u(x, y), y)
        + sp.sin(sp.diff(u(x, y), x) + sp.diff(u(x, y), y)),
    )
    res = pdesolve(eq, u(x, y), (x, y))
    assert res.method == "generalized_clairaut_complete_integral"
    raw = res.solution
    assert getattr(raw, "method", None) == "generalized_clairaut_complete_integral"
    sols = getattr(raw, "solutions", ())
    assert sols
    rhs = sols[0].rhs
    names = {str(s) for s in rhs.free_symbols}
    assert {"C1", "C2", "x", "y"}.issubset(names)


def test_explicit_generalized_clairaut_method():
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    eq = sp.Eq(
        x * sp.diff(u(x, y), x)
        + y * sp.diff(u(x, y), y)
        - u(x, y)
        + sp.sin(sp.diff(u(x, y), x) + sp.diff(u(x, y), y)),
        0,
    )
    res = pdesolve(eq, u(x, y), (x, y), method="generalized_clairaut_complete_integral")
    assert res.method == "generalized_clairaut_complete_integral"
    sols = res.solution.solutions
    assert len(sols) == 1
