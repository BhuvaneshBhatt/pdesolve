import sympy as sp

from pdesolve.classical import (
    PDEBoundaryCondition1D,
    canonicalize_pde_problem,
    construct_burgers_rarefaction,
    detect_scalar_conservation_law_family,
    pdesolve,
    recognize_pde_family,
    separate_variables_structured,
    solve_heat_equation_1d_half_line_transform,
    solve_heat_equation_1d_laplace_transform_formal,
    solve_heat_equation_1d_neumann_series,
    solve_inviscid_burgers_ivp_implicit,
    solve_wave_equation_1d_dirichlet_series,
    solve_wave_equation_1d_laplace_transform_formal,
    verify_pde_solution_with_data,
)


def test_family_recognizers_and_canonicalization():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")(x, t)
    eq = sp.Eq(2 * (sp.diff(u, t) + u * sp.diff(u, x) - 3 * sp.diff(u, x, 2)), 0)
    norm = canonicalize_pde_problem(eq, u, (x, t))
    fam = recognize_pde_family(norm, u, (x, t))
    assert fam is not None
    assert fam.family == "viscous_burgers"


def test_neumann_series_and_verification_smoke():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")(x, t)
    res = solve_heat_equation_1d_neumann_series(u, x=x, t=t, initial_profile=1, terms=3)
    assert res.method == "heat_neumann_cosine_series"
    report = verify_pde_solution_with_data(
        sp.Eq(sp.diff(u, t), sp.diff(u, x, 2)),
        res.solution,
        u,
        (x, t),
        ics={"initial_profile": 1},
        bcs=[PDEBoundaryCondition1D(0, "neumann", 0), PDEBoundaryCondition1D(sp.pi, "neumann", 0)],
    )
    assert report.verified


def test_half_line_and_laplace_transform_helpers():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")(x, t)
    half = solve_heat_equation_1d_half_line_transform(
        u, x=x, t=t, initial_profile=sp.exp(-x), boundary="dirichlet"
    )
    assert "half_line" in half.method
    wave = solve_wave_equation_1d_laplace_transform_formal(
        u, x=x, t=t, initial_displacement=sp.sin(x), initial_velocity=0
    )
    heat = solve_heat_equation_1d_laplace_transform_formal(u, x=x, t=t, initial_profile=sp.sin(x))
    assert isinstance(wave.solution, sp.Equality)
    assert isinstance(heat.solution, sp.Equality)


def test_wave_dirichlet_series_and_separation_of_variables():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")(x, t)
    res = solve_wave_equation_1d_dirichlet_series(
        u, x=x, t=t, initial_displacement=sp.sin(x), initial_velocity=0, terms=3
    )
    assert res.method == "wave_dirichlet_sine_series"
    sep = separate_variables_structured(
        sp.Eq(sp.diff(u, t), sp.diff(u, x, 2)),
        u,
        (x, t),
        bcs={"type": "dirichlet_homogeneous_interval", "length": sp.pi},
    )
    assert sep.basis_hint == "sine"


def test_conservation_and_burgers_helpers():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")(x, t)
    recog = detect_scalar_conservation_law_family(
        sp.Eq(sp.diff(u, t) + u * sp.diff(u, x), 0), u, (x, t)
    )
    assert recog.family == "inviscid_burgers"
    imp = solve_inviscid_burgers_ivp_implicit(u, x=x, t=t, initial_profile=sp.sin(x))
    assert imp.method == "inviscid_burgers_implicit_characteristics"
    rar = construct_burgers_rarefaction(0, 1, x=x, t=t)
    assert isinstance(rar, sp.Piecewise)


def test_auto_dispatch_new_methods():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")(x, t)
    # conservation-law path
    res1 = pdesolve(
        sp.Eq(sp.diff(u, t) + u * sp.diff(u, x), 0),
        u,
        (x, t),
        ics={"initial_profile": sp.sin(x)},
        method="conservation_law",
    )
    assert res1.method == "conservation_law_analysis"
    # structured separation path
    res2 = pdesolve(
        sp.Eq(sp.diff(u, t), sp.diff(u, x, 2)),
        u,
        (x, t),
        ics={"initial_profile": sp.sin(x)},
        bcs={"type": "dirichlet_homogeneous_interval", "length": sp.pi},
        method="separation_of_variables",
    )
    assert res2.method == "separation_of_variables"
