import sympy as sp

from pdesolve.classical import (
    classify_linear_second_order_pde,
    conserved_mass_statement,
    detect_conservation_law_1d,
    pdesolve,
    rankine_hugoniot_speed,
    separate_variables,
    solve_advection_equation_1d_fourier_transform,
    solve_heat_equation_1d_fourier_transform,
)


def test_classify_linear_second_order_with_assumptions():
    x, t = sp.symbols("x t", real=True)
    kappa = sp.symbols("kappa", positive=True)
    u = sp.Function("u")
    cls = classify_linear_second_order_pde(
        sp.Eq(sp.diff(u(x, t), t), kappa * sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        assumptions=sp.Q.positive(kappa),
    )
    assert cls.classification == "parabolic"


def test_separation_of_variables_heat_like():
    x, t = sp.symbols("x t", real=True)
    kappa = sp.symbols("kappa", positive=True)
    u = sp.Function("u")
    res = separate_variables(
        sp.Eq(sp.diff(u(x, t), t), kappa * sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        assumptions=sp.Q.positive(kappa),
    )
    assert res.method == "separation_heat_like"
    assert "Derivative(X(x), (x, 2))" in str(res.x_equation)
    assert "Derivative(T(t), t)" in str(res.t_equation)


def test_fourier_transform_methods():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    heat = solve_heat_equation_1d_fourier_transform(
        u(x, t), x=x, t=t, diffusivity=2, initial_profile=sp.exp(-(x**2))
    )
    adv = solve_advection_equation_1d_fourier_transform(
        u(x, t), x=x, t=t, speed=3, reaction=1, initial_profile=sp.exp(-(x**2))
    )
    assert heat.method == "fourier_heat_whole_line"
    assert adv.method == "fourier_advection_whole_line"


def test_conservation_law_detection_and_helpers():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    cons = detect_conservation_law_1d(
        sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t) ** 2 / 2, x), 0),
        u(x, t),
        (x, t),
    )
    assert sp.simplify(cons.flux - u(x, t) ** 2 / 2) == 0
    mass_stmt = conserved_mass_statement(cons)
    assert isinstance(mass_stmt, sp.Equality)
    usym = sp.Symbol("u")
    speed = rankine_hugoniot_speed(usym**2 / 2, 1, 3, u_symbol=usym)
    assert sp.simplify(speed - 2) == 0


def test_dispatcher_prefers_transform_and_conservation_analysis():
    x, t = sp.symbols("x t", real=True)
    kappa = sp.symbols("kappa", positive=True)
    u = sp.Function("u")

    heat = pdesolve(
        sp.Eq(sp.diff(u(x, t), t), kappa * sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics={"initial_profile": sp.exp(-(x**2))},
        prefer_transform=True,
        assumptions=sp.Q.positive(kappa),
    )
    assert heat.method == "fourier_heat_whole_line"

    cons = pdesolve(
        sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t) ** 2 / 2, x), 0),
        u(x, t),
        (x, t),
    )
    assert cons.method in {"pdsolve_first_order", "conservation_law_analysis"}
