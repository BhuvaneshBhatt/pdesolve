import sympy as sp

from pdesolve.classical import (
    build_quasilinear_characteristic_system_2vars_robust,
    detect_burgers_family,
    entropy_select_riemann_branch_scalar,
    extract_conservation_form_auto,
    pdesolve,
    solve_quasilinear_pde_characteristics_implicit,
    solve_scalar_conservation_law_riemann_general,
    solve_viscous_burgers_cole_hopf_formal,
)


def test_extract_conservation_form_auto_burgers():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)
    cf = extract_conservation_form_auto(eq, u, (x, t))
    assert sp.simplify(cf.flux - u(x, t) ** 2 / 2) == 0
    assert sp.simplify(cf.source) == 0


def test_detect_burgers_family_viscous():
    x, t, nu = sp.symbols("x t nu", positive=True, real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), nu * sp.diff(u(x, t), x, 2))
    fam = detect_burgers_family(eq, u, (x, t))
    assert fam is not None
    assert fam.family == "viscous_burgers"
    assert sp.simplify(fam.parameters["nu"] - nu) == 0


def test_general_riemann_solver_convex_flux():
    x, t, usym = sp.symbols("x t u", positive=True, real=True)
    flux = usym**2 / 2
    sel = entropy_select_riemann_branch_scalar(flux, 0, 1, u_symbol=usym)
    assert sel["branch"] == "rarefaction"
    res = solve_scalar_conservation_law_riemann_general(flux, 1, 0, x=x, t=t, u_symbol=usym)
    assert res.method == "scalar_conservation_riemann_shock"


def test_viscous_burgers_cole_hopf_formal():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")
    res = solve_viscous_burgers_cole_hopf_formal(
        u, x=x, t=t, viscosity=1, initial_profile=sp.Symbol("x0")
    )
    assert res.method == "viscous_burgers_cole_hopf_formal"
    assert isinstance(res.solution, sp.Equality)


def test_build_quasilinear_characteristic_system_robust():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)
    sys = build_quasilinear_characteristic_system_2vars_robust(
        eq, u, (x, t), initial_curve=(lambda xi: xi, 0), initial_data=lambda xi: xi
    )
    assert len(sys.odes) == 3
    assert sys.x_curve is not None


def test_quasilinear_implicit_solver_and_dispatch():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)
    res = solve_quasilinear_pde_characteristics_implicit(eq, u, (x, t), initial_profile=x**2)
    assert res.method == "quasilinear_implicit_characteristics"
    auto = pdesolve(eq, u, (x, t), method="quasilinear_implicit", ics={"initial_profile": x**2})
    assert auto.method == "quasilinear_implicit_characteristics"
