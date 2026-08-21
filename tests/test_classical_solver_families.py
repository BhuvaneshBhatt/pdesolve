import sympy as sp

from pdesolve.classical import (
    characteristic_form_first_order_2vars,
    classify_second_order_linear_pde_2vars,
    load_benchmark_measurement_csv,
    load_benchmark_measurement_json,
    pdesolve,
    solve_heat_equation_1d_dirichlet_series,
    solve_transport_ivp,
    solve_wave_equation_1d_ivp,
)


def test_benchmark_measurement_files_load():
    csv_rows = load_benchmark_measurement_csv().rows
    json_rows = load_benchmark_measurement_json().rows
    assert len(csv_rows) >= 10
    assert len(csv_rows) == len(json_rows)
    assert csv_rows[0]["PDE"] == json_rows[0]["PDE"]


def test_characteristic_data_form_detects_transport():
    x, t, a = sp.symbols("x t a", real=True)
    u = sp.Function("u")
    form = characteristic_form_first_order_2vars(
        sp.Eq(sp.diff(u(x, t), t) + a * sp.diff(u(x, t), x), 0), u(x, t)
    )
    assert sp.simplify(form.A - a) == 0
    assert sp.simplify(form.B - 1) == 0
    assert sp.simplify(form.C) == 0


def test_transport_ivp_constant_coefficients():
    x, t, a = sp.symbols("x t a", nonzero=True, real=True)
    u = sp.Function("u")
    res = solve_transport_ivp(
        sp.Eq(sp.diff(u(x, t), t) + a * sp.diff(u(x, t), x), 0),
        u(x, t),
        initial_profile=lambda z: z**2,
    )
    expected = sp.Eq(u(x, t), (-a * t + x) ** 2)
    assert sp.simplify(res.solution.rhs - expected.rhs) == 0


def test_second_order_type_classification_wave_and_heat():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    wave = classify_second_order_linear_pde_2vars(
        sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)), u(x, t)
    )
    heat = classify_second_order_linear_pde_2vars(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)), u(x, t)
    )
    assert wave.classification == "hyperbolic"
    assert heat.classification == "parabolic"


def test_wave_ivp_dalembert_contains_integral_term():
    x, t, c = sp.symbols("x t c", positive=True, real=True)
    u = sp.Function("u")
    res = solve_wave_equation_1d_ivp(
        u(x, t),
        x=x,
        t=t,
        wave_speed=c,
        initial_displacement=lambda z: sp.sin(z),
        initial_velocity=lambda s: s,
    )
    assert isinstance(res.solution, sp.Equality)
    assert "Integral" in str(res.solution.rhs)


def test_heat_dirichlet_series_solver_runs():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")
    res = solve_heat_equation_1d_dirichlet_series(
        u(x, t), x=x, t=t, initial_profile=lambda z: z * (sp.pi - z), terms=3
    )
    assert isinstance(res.solution, sp.Equality)
    assert "sin" in str(res.solution.rhs)


def test_auto_data_dispatch_prefers_transport_ivp_when_given_data():
    x, t, a = sp.symbols("x t a", nonzero=True, real=True)
    u = sp.Function("u")
    res = pdesolve(
        sp.Eq(sp.diff(u(x, t), t) + a * sp.diff(u(x, t), x), 0),
        u(x, t),
        ics={"initial_profile": lambda z: z + 1},
        method="transport_ivp",
    )
    assert sp.simplify(res.solution.rhs - (x - a * t + 1)) == 0
