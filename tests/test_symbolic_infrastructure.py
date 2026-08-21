import sympy as sp

from pdesolve import (
    ClassicalResidualVerifier,
    ClosedFormPDEResult,
    ImplicitPDEResult,
    ImplicitSolutionVerifier,
    SeriesPDEResult,
    SeriesVerifier,
    SturmLiouvilleProblem,
    TransformPDEResult,
    TransformVerifier,
    analyze_lie_point_symmetries,
    evaluate_inner_transforms,
    invariants_from_point_generator,
    select_verification_strategy,
    separate_product_pde,
    solve_regular_constant_sturm_liouville,
    verify_result,
)


def test_verification_strategy_selection_by_representation():
    x = sp.symbols("x")
    assert isinstance(
        select_verification_strategy(ClosedFormPDEResult("closed", sp.Eq(sp.Symbol("u"), x))),
        ClassicalResidualVerifier,
    )
    assert isinstance(
        select_verification_strategy(ImplicitPDEResult("implicit", sp.Eq(sp.Symbol("F"), 0))),
        ImplicitSolutionVerifier,
    )
    assert isinstance(
        select_verification_strategy(SeriesPDEResult("series", sp.Integer(0))), SeriesVerifier
    )
    assert isinstance(
        select_verification_strategy(
            TransformPDEResult(
                "fourier_transform", sp.Integral(sp.exp(-(x**2)), (x, -sp.oo, sp.oo))
            )
        ),
        TransformVerifier,
    )


def test_result_verification_classical_residual():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    result = ClosedFormPDEResult("transport", sp.Eq(u(x, t), x - t))
    report = verify_result(
        sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0), result, u(x, t), (x, t)
    )
    assert report.pde_verified is True


def test_general_product_separation_derives_heat_odes():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    sep = separate_product_pde(sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)), u(x, t), (x, t))
    assert sep.verified_separable
    assert len(sep.factor_equations) == 2
    assert sep.factor_equations[0].lhs.has(sp.Derivative)
    assert sep.factor_equations[1].lhs.has(sp.Derivative)


def test_general_product_separation_rejects_coupled_nonlinearity():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    try:
        separate_product_pde(
            sp.Eq(sp.diff(u(x, t), t), u(x, t) * sp.diff(u(x, t), x)), u(x, t), (x, t)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("nonseparable nonlinear PDE should be rejected")


def test_sturm_liouville_dirichlet_spectrum_and_projection():
    x, L = sp.symbols("x L", positive=True)
    X = sp.Function("X")(x)
    problem = SturmLiouvilleProblem(x, X, 1, 0, 1, (0, L))
    spec = solve_regular_constant_sturm_liouville(problem)
    n = spec.index
    assert sp.simplify(spec.eigenvalues - (sp.pi * n / L) ** 2) == 0
    assert sp.simplify(spec.eigenfunctions.subs(x, 0)) == 0
    assert sp.simplify(spec.eigenfunctions.subs(x, L)) == 0
    coeff = spec.coefficient(sp.sin(sp.pi * x / L))
    assert sp.simplify(coeff.subs(n, 1) - 1) == 0


def test_sturm_liouville_neumann_zero_mode():
    x = sp.symbols("x")
    X = sp.Function("X")(x)
    spec = solve_regular_constant_sturm_liouville(
        SturmLiouvilleProblem(x, X, 1, 0, 1, (0, sp.pi), "neumann", "neumann")
    )
    assert spec.includes_zero_mode
    assert sp.simplify(spec.eigenvalues.subs(spec.index, 0)) == 0


def test_transform_postprocessing_evaluates_only_inner_profile_transform():
    x, xi, w, t = sp.symbols("x xi w t", positive=True)
    inner = sp.Integral(sp.exp(-xi) * sp.sin(w * xi), (xi, 0, sp.oo))
    outer = sp.Integral(sp.exp(-(w**2) * t) * sp.sin(w * x) * inner, (w, 0, sp.oo))
    processed, report = evaluate_inner_transforms(outer)
    assert report.changed
    assert not processed.has(inner)
    assert processed.has(sp.Integral)  # outer inversion intentionally remains formal


def test_lie_analysis_generates_determining_equations_and_generators():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    analysis = analyze_lie_point_symmetries(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)), u(x, t), (x, t), polynomial_degree=1
    )
    assert analysis.determining_equations
    assert analysis.generators
    # translations in x and t are among the affine-polynomial heat symmetries
    assert any(g[0] == (1, 0) and g[1] == 0 for g in analysis.generators)
    assert any(g[0] == (0, 1) and g[1] == 0 for g in analysis.generators)


def test_lie_analysis_invariants_for_traveling_wave_generator():
    x, t, U = sp.symbols("x t U")
    inv = invariants_from_point_generator((x, t), U, (1, 1), 0)
    assert any(
        sp.simplify(z - (x - t)) == 0 or sp.simplify(z + (x - t)) == 0 for z in inv.invariants
    )
    assert U in inv.invariants
    assert sp.simplify(inv.jacobian) != 0
