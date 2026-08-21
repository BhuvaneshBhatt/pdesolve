from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import sympy as sp
from sympy.assumptions import assuming
from sympy.solvers.pde import classify_pde, pdsolve

from .classical_beam import (
    SpectralPDEResult as SpectralPDEResult,
)
from .classical_beam import (
    solve_simply_supported_beam_ibvp as solve_simply_supported_beam_ibvp,
)
from .classical_classification import (
    LinearSecondOrderPDEClassification as LinearSecondOrderPDEClassification,
)
from .classical_classification import (
    SecondOrderLinearType2D as SecondOrderLinearType2D,
)
from .classical_classification import (
    classify_linear_second_order_pde as classify_linear_second_order_pde,
)
from .classical_classification import (
    classify_second_order_linear_pde_2vars as classify_second_order_linear_pde_2vars,
)
from .classical_evolution import (
    solve_heat_equation_1d_dirichlet_series as solve_heat_equation_1d_dirichlet_series,
)
from .classical_evolution import (
    solve_heat_equation_1d_whole_line_ivp as solve_heat_equation_1d_whole_line_ivp,
)
from .classical_evolution import (
    solve_wave_equation_1d_ivp as solve_wave_equation_1d_ivp,
)
from .classical_first_order import (
    FirstOrderCharacteristicForm as FirstOrderCharacteristicForm,
)
from .classical_first_order import (
    PDEIVPResult as PDEIVPResult,
)
from .classical_first_order import (
    characteristic_form_first_order_2vars as characteristic_form_first_order_2vars,
)
from .classical_first_order import (
    solve_first_order_pde_characteristic as solve_first_order_pde_characteristic,
)
from .classical_first_order import (
    solve_transport_ivp as solve_transport_ivp,
)
from .classical_separation import (
    SeparationOfVariablesResult as SeparationOfVariablesResult,
)
from .classical_separation import (
    separate_variables as separate_variables,
)
from .classical_symbolic_helpers import (
    _as_zero_expr,
    _dep_and_vars,
    _safe_sub_profile,
    _safe_sub_profile_general,
)
from .classical_transforms import (
    TransformMethodResult as TransformMethodResult,
)
from .classical_transforms import (
    solve_advection_equation_1d_fourier_transform as solve_advection_equation_1d_fourier_transform,
)
from .classical_transforms import (
    solve_heat_equation_1d_fourier_transform as solve_heat_equation_1d_fourier_transform,
)
from .constant_coeff import (
    PDEGeneralSolutionResult,
    _polynomial_particular,
)
from .constant_coeff import (
    detect_linear_constant_coefficient_pde as detect_linear_constant_coefficient_pde,
)
from .constant_coeff import (
    pdesolve_constant_coefficient as solve_linear_constant_coefficient_pde,
)
from .family_recognizers import (
    PDEBoundaryCondition1D as PDEBoundaryCondition1D,
)
from .family_recognizers import (
    PDEInitialCondition1D as PDEInitialCondition1D,
)
from .family_recognizers import (
    PDEVerificationReport,
    SeparationResult,
    _canonical_linear_pde_1d_xt,
    _normalize_condition_dicts,
)
from .family_recognizers import (
    detect_burgers_family as detect_burgers_family,
)
from .family_recognizers import (
    detect_scalar_conservation_law_family as detect_scalar_conservation_law_family,
)
from .family_recognizers import (
    recognize_pde_family as recognize_pde_family,
)


def particular_solution_polynomial_rhs_inverse_operator(ccpde, rhs=None):
    rhs = ccpde.rhs if rhs is None else rhs
    return _polynomial_particular(ccpde, sp.expand(rhs)).solution


@dataclass(frozen=True)
class ConservationLaw1D:
    indep_vars: tuple[sp.Symbol, sp.Symbol]
    dep_function: sp.Expr
    density: sp.Expr
    flux: sp.Expr
    normalized_equation: sp.Equality
    details: dict


def detect_conservation_law_1d(eq_or_expr, dep_expr_or_func, indep_vars=None):
    """
    Detect a scalar 1D conservation law of the form
        D_t rho(x,t,u) + D_x F(x,t,u) = 0

    Current starter implementation targets the common scalar form
        u_t + d/dx Phi(x,t,u) = 0
    i.e. density rho = u.
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if len(vars_) != 2:
        raise ValueError("Expected two variables (x,t).")
    x, t = vars_
    zero = _as_zero_expr(eq_or_expr)
    ux = sp.diff(uexpr, x)
    ut = sp.diff(uexpr, t)

    # Require first order and linear in ux, ut
    for node in sp.preorder_traversal(zero):
        if isinstance(node, sp.Derivative) and node.expr == uexpr:
            total_order = sum(count for _, count in node.variable_count)
            if total_order > 1:
                raise NotImplementedError(
                    "Current conservation-law detector is for first-order scalar PDEs."
                )

    try:
        poly = sp.Poly(zero, ux, ut, domain="EX")
    except Exception as exc:
        raise ValueError("Could not treat PDE as polynomial in first derivatives.") from exc
    if poly.total_degree() > 1:
        raise NotImplementedError(
            "Current conservation-law detector expects linear dependence on first derivatives."
        )

    A = sp.expand(sp.diff(zero, ut))
    B = sp.expand(sp.diff(zero, ux))
    rest = sp.expand(zero.subs({ut: 0, ux: 0}))

    if sp.simplify(A - 1) != 0:
        raise NotImplementedError("Current detector expects unit density coefficient on u_t.")

    # Need B = ∂Phi/∂u and rest = ∂Phi/∂x with u held constant.
    usym = sp.Symbol("_ucons")
    B_hold = sp.expand(B.subs({uexpr: usym}))
    rest_hold = sp.expand(rest.subs({uexpr: usym}))
    Phi_u_hold = sp.integrate(B_hold, usym)
    partial_x_hold = sp.diff(Phi_u_hold, x)
    correction = sp.simplify(rest_hold - partial_x_hold)
    if correction.has(usym):
        raise NotImplementedError("Could not reconstruct a scalar flux independent of path in u.")
    Phi_hold = sp.expand(Phi_u_hold + sp.integrate(correction, x))
    Phi_expr = sp.expand(Phi_hold.subs({usym: uexpr}))
    check = sp.simplify(
        sp.diff(Phi_hold, x).subs({usym: uexpr})
        + sp.diff(Phi_hold, usym).subs({usym: uexpr}) * ux
        - (B * ux + rest)
    )
    if check != 0:
        raise NotImplementedError("Could not verify reconstructed scalar flux.")

    return ConservationLaw1D(
        indep_vars=(x, t),
        dep_function=uexpr,
        density=uexpr,
        flux=Phi_expr,
        normalized_equation=sp.Eq(ut + sp.diff(Phi_expr, x) + sp.diff(Phi_expr, uexpr) * ux, 0),
        details={"type": "scalar_conservation_law", "rho": uexpr, "flux": Phi_expr},
    )


def rankine_hugoniot_speed(flux, u_left, u_right, u_symbol=None):
    """
    Rankine-Hugoniot shock speed for u_t + (f(u))_x = 0.
    """
    u = sp.Symbol("u") if u_symbol is None else sp.sympify(u_symbol)
    f = sp.sympify(flux)
    return sp.simplify((f.subs(u, u_right) - f.subs(u, u_left)) / (u_right - u_left))


def conserved_mass_statement(conslaw):
    """
    Formal whole-line conserved mass statement for rho_t + F_x = 0.
    """
    x, t = conslaw.indep_vars
    mass = sp.Integral(conslaw.density, (x, -sp.oo, sp.oo))
    return sp.Eq(sp.diff(mass, t), 0)


def with_parameter_assumptions(assumptions, fn, *args, **kwargs):
    """Evaluate a callable under SymPy assumptions and refine the result when possible."""
    if assumptions in (True, None):
        return fn(*args, **kwargs)
    with assuming(assumptions):
        res = fn(*args, **kwargs)
    return res


@dataclass(frozen=True)
class FirstOrderLinearForm2D:
    indep_vars: tuple[sp.Symbol, sp.Symbol]
    dep_function: sp.Expr
    A: sp.Expr
    B: sp.Expr
    D: sp.Expr
    E: sp.Expr
    normalized_equation: sp.Equality
    is_constant_coefficient: bool


@dataclass(frozen=True)
class PDEProblemProfile:
    indep_vars: tuple[sp.Symbol, ...]
    dep_function: sp.Expr
    normalized_equation: sp.Equality
    zero_expression: sp.Expr
    order: int
    principal_solved_form: object | None
    characteristic_data: FirstOrderCharacteristicForm | None
    first_order_linear: FirstOrderLinearForm2D | None
    second_order_class: LinearSecondOrderPDEClassification | None
    canonical_family: str | None
    conservation_law: ConservationLaw1D | None
    details: dict


@dataclass(frozen=True)
class PDESolverMethodCandidate:
    method: str
    score: int
    reasons: tuple[str, ...]
    details: dict


def _infer_pde_order(zero_expr, uexpr):
    order = 0
    for node in sp.preorder_traversal(zero_expr):
        if isinstance(node, sp.Derivative) and node.expr == uexpr:
            order = max(order, sum(count for _, count in node.variable_count))
    return order


def detect_first_order_linear_form_2vars(
    eq_or_expr, dep_expr_or_func, indep_vars=None
) -> FirstOrderLinearForm2D:
    """
    Detect a broad two-variable linear first-order PDE of the form
        A(x,y) u_x + B(x,y) u_y + D(x,y) u = E(x,y)
    where A,B,D,E may vary with the independent variables but not with u.
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if len(vars_) != 2:
        raise ValueError("Expected two independent variables.")
    x, y = vars_
    zero = _as_zero_expr(eq_or_expr)
    ux = sp.diff(uexpr, x)
    uy = sp.diff(uexpr, y)

    for node in sp.preorder_traversal(zero):
        if isinstance(node, sp.Derivative) and node.expr == uexpr:
            total_order = sum(count for _, count in node.variable_count)
            if total_order > 1:
                raise ValueError("Equation is not first-order.")

    try:
        poly = sp.Poly(zero, ux, uy, uexpr, domain="EX")
    except Exception as exc:
        raise ValueError("Could not treat PDE as polynomial in first derivatives and u.") from exc
    if poly.total_degree() > 1:
        raise ValueError("Equation is not linear in first derivatives and u.")

    A = sp.expand(sp.diff(zero, ux))
    B = sp.expand(sp.diff(zero, uy))
    D = sp.expand(sp.diff(zero.subs({ux: 0, uy: 0}), uexpr))
    E = sp.expand(-(zero.subs({ux: 0, uy: 0, uexpr: 0})))

    test = sp.expand(zero - A * ux - B * uy - D * uexpr + E)
    if test != 0:
        raise ValueError("Could not isolate a linear first-order form.")

    for coeff in (A, B, D, E):
        if not coeff.free_symbols.isdisjoint({uexpr, ux, uy}):
            raise ValueError("Coefficients still depend on u or its first derivatives.")

    normalized = sp.Eq(sp.expand(A * ux + B * uy + D * uexpr), sp.expand(E))
    is_const = all(coeff.free_symbols.isdisjoint({x, y}) for coeff in (A, B, D, E))
    return FirstOrderLinearForm2D((x, y), uexpr, A, B, D, E, normalized, is_const)


def solve_first_order_linear_pde_pdsolve(eq_or_expr, dep_expr_or_func, indep_vars=None):
    form = detect_first_order_linear_form_2vars(eq_or_expr, dep_expr_or_func, indep_vars)
    try:
        hints = classify_pde(form.normalized_equation)
    except Exception:
        hints = ()
    try:
        sol = pdsolve(form.normalized_equation)
        return PDEIVPResult(
            method="pdsolve_first_order_linear",
            solution=sol,
            details={"classification_hints": tuple(hints), "linear_form": form},
        )
    except Exception as exc:
        raise NotImplementedError(
            "Could not solve the linear first-order PDE with pdsolve."
        ) from exc


def _reduce_and_solve_by_symmetry(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True, max_symmetry_steps=2
):
    """Best-effort symmetry-first solve hook that reduces and then solves the reduced equation."""
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    from .jet_space import (
        build_scalar_general_solved_pde_from_equation,
        build_scalar_jet_equation_from_sympy_pde,
    )
    from .workflows import repeated_reduction_workflow_scalar_kd_frobenius_default

    jet, pde = build_scalar_jet_equation_from_sympy_pde(
        vars_,
        uexpr.func,
        sp.Eq(_as_zero_expr(eq_or_expr), 0),
        max_order=max(3, _infer_pde_order(_as_zero_expr(eq_or_expr), uexpr) or 1),
        dep_name=getattr(uexpr.func, "__name__", "u"),
    )
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(
        jet, pde, max_principal_order=max(3, jet.max_order)
    )
    workflow = repeated_reduction_workflow_scalar_kd_frobenius_default(
        eq_obj, max_steps=max_symmetry_steps
    )
    final_eq = workflow.final_equation
    if final_eq is None:
        raise NotImplementedError("Symmetry workflow did not produce a reduced equation.")
    solved = solve_reduced_equation_auto(
        final_eq, assumptions=assumptions, max_symmetry_steps=max_symmetry_steps
    )
    return PDEIVPResult(
        method="symmetry_reduction_plus_postsolve",
        solution=solved.solution,
        details={"workflow": workflow, "postsolve": solved},
    )


def solve_heat_equation_1d_neumann_series(
    dep_expr_or_func, *, x=None, t=None, diffusivity=1, length=sp.pi, initial_profile=None, terms=6
):
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    x, t = vars_
    kappa = sp.sympify(diffusivity)
    L = sp.sympify(length)
    if initial_profile is None:
        raise ValueError("initial_profile is required.")
    xi = sp.Symbol("xi", real=True)
    if callable(initial_profile):
        phi = initial_profile(xi)
    else:
        expr = sp.sympify(initial_profile)
        free = list(expr.free_symbols)
        phi = expr.subs(free[0], xi) if len(free) == 1 else expr
    a0 = sp.Integral(phi, (xi, 0, L)) / L
    series = a0
    for n in range(1, int(terms) + 1):
        an = 2 / L * sp.Integral(phi * sp.cos(n * sp.pi * xi / L), (xi, 0, L))
        series += an * sp.cos(n * sp.pi * x / L) * sp.exp(-kappa * (n * sp.pi / L) ** 2 * t)
    return PDEIVPResult(
        method="heat_neumann_cosine_series",
        solution=sp.Eq(uexpr, sp.simplify(series)),
        details={"diffusivity": kappa, "length": L, "terms": int(terms)},
    )


def solve_heat_equation_1d_half_line_transform(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    diffusivity=1,
    initial_profile=None,
    boundary="dirichlet",
    fourier_variable=None,
):
    """Return the Fourier sine/cosine transform solution on ``x > 0``.

    The transform is intentionally kept unevaluated.  In particular, do not run
    generic ``simplify`` on the nested improper integral: SymPy may attempt to
    evaluate or transform the integrals, which can turn construction of a formal
    solution into a very expensive symbolic integration problem.  Evaluation of
    the profile transform or inverse transform is a separate post-processing step.
    """
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    x, t = vars_
    kappa = sp.sympify(diffusivity)
    w = (
        sp.Symbol("w", positive=True, real=True)
        if fourier_variable is None
        else sp.sympify(fourier_variable)
    )
    xi = sp.Symbol("xi", positive=True, real=True)
    if initial_profile is None:
        raise ValueError("initial_profile is required.")
    if callable(initial_profile):
        phi = initial_profile(xi)
    else:
        expr = sp.sympify(initial_profile)
        free = list(expr.free_symbols)
        phi = expr.subs(free[0], xi) if len(free) == 1 else expr

    boundary_key = str(boundary).strip().lower()
    if boundary_key not in {"dirichlet", "neumann"}:
        raise ValueError("boundary must be 'dirichlet' or 'neumann'.")
    trig = sp.sin if boundary_key == "dirichlet" else sp.cos

    # Write the spectral transform explicitly instead of embedding all factors in
    # one nested kernel.  This is mathematically identical, easier to inspect, and
    # prevents generic simplification from trying to solve the improper integrals.
    profile_transform = sp.Integral(phi * trig(w * xi), (xi, 0, sp.oo))
    sol = (
        sp.Integer(2)
        / sp.pi
        * sp.Integral(
            sp.exp(-kappa * w**2 * t) * trig(w * x) * profile_transform,
            (w, 0, sp.oo),
        )
    )
    return PDEIVPResult(
        method=f"heat_half_line_{boundary_key}_transform",
        solution=sp.Eq(uexpr, sol),
        details={
            "diffusivity": kappa,
            "boundary": boundary_key,
            "transform_variable": w,
            "profile_transform": profile_transform,
            "transform_evaluated": False,
        },
    )


def solve_wave_equation_1d_dirichlet_series(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    wave_speed=1,
    length=sp.pi,
    initial_displacement=None,
    initial_velocity=None,
    terms=6,
):
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    x, t = vars_
    c = sp.sympify(wave_speed)
    L = sp.sympify(length)
    if initial_displacement is None:
        raise ValueError("initial_displacement is required.")
    if initial_velocity is None:

        def initial_velocity(_):
            return 0

    xi = sp.Symbol("xi", real=True)

    def _profile(profile):
        if callable(profile):
            return profile(xi)
        expr = sp.sympify(profile)
        free = list(expr.free_symbols)
        return expr.subs(free[0], xi) if len(free) == 1 else expr

    f = _profile(initial_displacement)
    g = _profile(initial_velocity)
    series = 0
    for n in range(1, int(terms) + 1):
        bn = 2 / L * sp.Integral(f * sp.sin(n * sp.pi * xi / L), (xi, 0, L))
        cn = 2 / (c * n * sp.pi) * sp.Integral(g * sp.sin(n * sp.pi * xi / L), (xi, 0, L))
        series += (
            bn * sp.cos(c * n * sp.pi * t / L) + L * cn * sp.sin(c * n * sp.pi * t / L)
        ) * sp.sin(n * sp.pi * x / L)
    return PDEIVPResult(
        method="wave_dirichlet_sine_series",
        solution=sp.Eq(uexpr, sp.simplify(series)),
        details={"wave_speed": c, "length": L, "terms": int(terms)},
    )


def solve_wave_equation_1d_laplace_transform_formal(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    wave_speed=1,
    initial_displacement=None,
    initial_velocity=None,
    laplace_variable=None,
):
    _, vars_ = _dep_and_vars(dep_expr_or_func, (x, t) if x is not None and t is not None else None)
    x, t = vars_
    s = (
        sp.Symbol("s", positive=True, real=True)
        if laplace_variable is None
        else sp.sympify(laplace_variable)
    )
    c = sp.sympify(wave_speed)
    U = sp.Function("U")(x, s)
    f = (
        sp.sympify(0)
        if initial_displacement is None
        else (
            initial_displacement(x)
            if callable(initial_displacement)
            else sp.sympify(initial_displacement).subs(
                list(sp.sympify(initial_displacement).free_symbols)[0], x
            )
            if isinstance(initial_displacement, sp.Expr)
            and len(sp.sympify(initial_displacement).free_symbols) == 1
            else sp.sympify(initial_displacement)
        )
    )
    g = (
        sp.sympify(0)
        if initial_velocity is None
        else (
            initial_velocity(x)
            if callable(initial_velocity)
            else sp.sympify(initial_velocity).subs(
                list(sp.sympify(initial_velocity).free_symbols)[0], x
            )
            if isinstance(initial_velocity, sp.Expr)
            and len(sp.sympify(initial_velocity).free_symbols) == 1
            else sp.sympify(initial_velocity)
        )
    )
    ode = sp.Eq(c**2 * sp.diff(U, x, 2) - s**2 * U, -(s * f + g))
    return PDEIVPResult(
        method="laplace_transform_wave_formal",
        solution=ode,
        details={"laplace_variable": s, "wave_speed": c},
    )


def solve_heat_equation_1d_laplace_transform_formal(
    dep_expr_or_func, *, x=None, t=None, diffusivity=1, initial_profile=None, laplace_variable=None
):
    _, vars_ = _dep_and_vars(dep_expr_or_func, (x, t) if x is not None and t is not None else None)
    x, t = vars_
    s = (
        sp.Symbol("s", positive=True, real=True)
        if laplace_variable is None
        else sp.sympify(laplace_variable)
    )
    kappa = sp.sympify(diffusivity)
    U = sp.Function("U")(x, s)
    f = (
        sp.sympify(0)
        if initial_profile is None
        else (
            initial_profile(x)
            if callable(initial_profile)
            else sp.sympify(initial_profile).subs(
                list(sp.sympify(initial_profile).free_symbols)[0], x
            )
            if isinstance(initial_profile, sp.Expr)
            and len(sp.sympify(initial_profile).free_symbols) == 1
            else sp.sympify(initial_profile)
        )
    )
    ode = sp.Eq(kappa * sp.diff(U, x, 2) - s * U, -f)
    return PDEIVPResult(
        method="laplace_transform_heat_formal",
        solution=ode,
        details={"laplace_variable": s, "diffusivity": kappa},
    )


def separate_variables_structured(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True, bcs=None
):
    """Deeper separation helper with basis hints from boundary data."""
    sep = separate_variables(eq_or_expr, dep_expr_or_func, indep_vars, assumptions=assumptions)
    _, norm_bcs = _normalize_condition_dicts(None, bcs)
    basis_hint = None
    if len(norm_bcs) == 2:
        kinds = {bc.kind for bc in norm_bcs}
        vals = [sp.simplify(bc.value if not callable(bc.value) else bc.value(0)) for bc in norm_bcs]
        if kinds == {"dirichlet"} and all(v == 0 for v in vals):
            basis_hint = "sine"
        elif kinds == {"neumann"} and all(v == 0 for v in vals):
            basis_hint = "cosine"
    family = "generic_separable"
    det = _canonical_linear_pde_1d_xt(
        eq_or_expr, dep_expr_or_func, indep_vars, assumptions=assumptions
    )
    if det is not None:
        family = det["kind"]
    return SeparationResult(
        family=family,
        ansatz=sep.ansatz,
        separated_odes=(sep.x_equation, sep.t_equation),
        separation_constants=(sep.separation_constant,),
        basis_hint=basis_hint,
        details={"base": sep, "boundary_conditions": norm_bcs},
    )


def solve_inviscid_burgers_ivp_implicit(dep_expr_or_func, *, x=None, t=None, initial_profile=None):
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    x, t = vars_
    if initial_profile is None:
        raise ValueError("initial_profile is required.")
    xi = sp.Symbol("xi", real=True)
    g = (
        initial_profile(xi)
        if callable(initial_profile)
        else sp.sympify(initial_profile).subs(list(sp.sympify(initial_profile).free_symbols)[0], xi)
        if isinstance(initial_profile, sp.Expr)
        and len(sp.sympify(initial_profile).free_symbols) == 1
        else sp.sympify(initial_profile)
    )
    implicit = sp.Eq(uexpr, g.subs(xi, x - t * uexpr))
    return PDEIVPResult(
        method="inviscid_burgers_implicit_characteristics",
        solution=implicit,
        details={"characteristic_foot": x - t * uexpr},
    )


def construct_burgers_rarefaction(u_left, u_right, *, x=None, t=None):
    x = sp.Symbol("x", real=True) if x is None else sp.sympify(x)
    t = sp.Symbol("t", positive=True, real=True) if t is None else sp.sympify(t)
    xi = sp.simplify(x / t)
    return sp.Piecewise(
        (u_left, xi <= u_left), (xi, sp.And(xi >= u_left, xi <= u_right)), (u_right, True)
    )


def verify_pde_solution_with_data(
    eq_or_expr,
    solution_eq,
    dep_expr_or_func=None,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    assumptions=True,
):
    zero = _as_zero_expr(eq_or_expr)
    if not isinstance(solution_eq, sp.Equality):
        raise ValueError("solution_eq must be an Equality.")
    lhs = solution_eq.lhs
    rhs = solution_eq.rhs
    if dep_expr_or_func is None:
        if isinstance(lhs, sp.Expr) and getattr(lhs, "is_Function", False):
            dep = lhs
            vars_ = lhs.args
        else:
            raise ValueError("Could not infer dependent function.")
    else:
        dep, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    sub_eq = {dep: rhs}
    # Replace derivatives explicitly.
    expr = zero
    for node in list(sp.preorder_traversal(expr)):
        if isinstance(node, sp.Derivative) and node.expr == dep:
            deriv = rhs
            for var, count in node.variable_count:
                deriv = sp.diff(deriv, var, count)
            expr = expr.xreplace({node: deriv})
    expr = sp.expand(expr.subs(sub_eq))
    pde_resid = sp.simplify(expr)

    norm_ics, norm_bcs = _normalize_condition_dicts(ics, bcs)
    initial_residuals = []
    boundary_residuals = []
    if len(vars_) >= 2:
        x = vars_[0]
        t = vars_[1]
        for ic in norm_ics:
            val = rhs.subs(t, ic.location)
            if ic.derivative_order == 1:
                val = sp.diff(rhs, t).subs(t, ic.location)
            target = ic.value(x) if callable(ic.value) else sp.sympify(ic.value)
            initial_residuals.append(sp.simplify(val - target))
        for bc in norm_bcs:
            if bc.kind == "dirichlet":
                val = rhs.subs(x, bc.location)
                target = bc.value(t) if callable(bc.value) else sp.sympify(bc.value)
            elif bc.kind == "neumann":
                val = sp.diff(rhs, x).subs(x, bc.location)
                target = bc.value(t) if callable(bc.value) else sp.sympify(bc.value)
            else:
                val = (sp.diff(rhs, x) + sp.sympify(bc.coefficient) * rhs).subs(x, bc.location)
                target = bc.value(t) if callable(bc.value) else sp.sympify(bc.value)
            boundary_residuals.append(sp.simplify(val - target))
    verified = sp.simplify(pde_resid) == 0 and all(
        sp.simplify(r) == 0 for r in initial_residuals + boundary_residuals
    )
    return PDEVerificationReport(
        pde_residual=pde_resid,
        initial_residuals=tuple(initial_residuals),
        boundary_residuals=tuple(boundary_residuals),
        assumptions=assumptions,
        verified=verified,
    )


@dataclass(frozen=True)
class PDEBoundarySpec:
    variable: sp.Symbol
    location: sp.Expr
    kind: str
    value: sp.Expr | Callable
    side: str | None = None


@dataclass(frozen=True)
class PDEInitialSpec:
    variables: tuple[sp.Symbol, ...]
    manifold: tuple[tuple[sp.Symbol, sp.Expr], ...]
    value: sp.Expr | Callable
    derivative_order: int = 0


@dataclass(frozen=True)
class PDETransformDomainProblem:
    transform_type: str
    transformed_equation: sp.Equality
    transformed_unknown: sp.Expr
    inversion_hint: str
    details: dict


@dataclass(frozen=True)
class PDESolutionPlan:
    profile: object
    steps: tuple[PDESolverMethodCandidate, ...]
    details: dict


@dataclass(frozen=True)
class PDESolutionRecord:
    method: str
    solution: sp.Expr | sp.Equality
    steps: tuple[str, ...]
    verification: dict
    assumptions: object
    canonical_equation: sp.Equality
    details: dict


def normalize_problem_data(
    ics=None, bcs=None, *, indep_vars=None, assumptions=True, domain=None, dep_expr=None
):
    from .conditions import parse_conditions
    from .domains import infer_domain_geometry

    norm_ics, norm_bcs = _normalize_condition_dicts(ics, bcs)
    cond_model = parse_conditions(ics, bcs, dep_expr=dep_expr, indep_vars=tuple(indep_vars or ()))

    extra_bcs = []
    if isinstance(bcs, dict):
        btype = bcs.get("type")
        if btype == "dirichlet_rectangle":
            x0 = bcs.get("x0", 0)
            x1 = bcs.get("x1", sp.pi)
            extra_bcs.extend(
                [
                    PDEBoundaryCondition1D(x0, "dirichlet", bcs.get("left", 0)),
                    PDEBoundaryCondition1D(x1, "dirichlet", bcs.get("right", 0)),
                ]
            )
            # retain transverse metadata on the domain instead of forcing 1D BC objects for y-sides
        elif btype == "robin_interval":
            L = bcs.get("length", sp.pi)
            left = bcs.get("left", (1, 0))
            right = bcs.get("right", (1, 0))
            lcoef = left[0] if isinstance(left, (tuple, list)) else 1
            rcoef = right[0] if isinstance(right, (tuple, list)) else 1
            lval = left[1] if isinstance(left, (tuple, list)) and len(left) > 1 else 0
            rval = right[1] if isinstance(right, (tuple, list)) and len(right) > 1 else 0
            extra_bcs.extend(
                [
                    PDEBoundaryCondition1D(0, "robin", lval, coefficient=lcoef),
                    PDEBoundaryCondition1D(L, "robin", rval, coefficient=rcoef),
                ]
            )
        elif btype == "dirichlet_half_line":
            extra_bcs.append(PDEBoundaryCondition1D(0, "dirichlet", bcs.get("boundary_value", 0)))
        elif btype == "neumann_half_line":
            extra_bcs.append(PDEBoundaryCondition1D(0, "neumann", bcs.get("boundary_value", 0)))

    dom = _normalize_domain(domain, bcs=bcs, ics=ics)
    inferred = infer_domain_geometry(
        indep_vars=tuple(indep_vars or ()), bcs=bcs, condition_model=cond_model
    )
    if dom is None:
        dom = PDEProblemDomain(
            geometry=inferred.kind,
            spatial_interval=inferred.extents.get("x"),
            metadata={"geometry": inferred},
        )
    else:
        dom = PDEProblemDomain(
            geometry=dom.geometry,
            spatial_interval=dom.spatial_interval,
            time_interval=dom.time_interval,
            metadata={**(dom.metadata or {}), "geometry": inferred},
        )

    condition_residuals = []
    try:
        if cond_model.initial_conditions and cond_model.boundary_conditions:
            for ic in cond_model.initial_conditions:
                for bc in cond_model.boundary_conditions:
                    if ic.location == bc.location:
                        condition_residuals.append(sp.Integer(0))
    except Exception:
        pass
    return PDEProblemData(
        domain=dom,
        initial_conditions=tuple(norm_ics),
        boundary_conditions=tuple(norm_bcs) + tuple(extra_bcs),
        condition_residuals=tuple(condition_residuals),
        conditions_satisfied=all(sp.simplify(r) == 0 for r in condition_residuals)
        if condition_residuals
        else True,
    )


@dataclass(frozen=True)
class CanonicalPDEProblem:
    equation: sp.Equality
    overall_factor: sp.Expr
    signature: tuple
    details: dict


def pde_problem_equivalence_signature(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True
):
    can = canonicalize_pde_problem(
        eq_or_expr, dep_expr_or_func, indep_vars, assumptions=assumptions
    )
    return can.signature


def verify_solution_record(
    eq_or_expr, solution, dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None, assumptions=True
):
    try:
        res = verify_pde_solution_with_data(
            eq_or_expr,
            solution,
            dep_expr_or_func,
            indep_vars,
            ics=ics,
            bcs=bcs,
            assumptions=assumptions,
        )
        return res
    except Exception as exc:
        return {"verified": False, "error": str(exc)}


def build_quasilinear_characteristic_system_2vars(eq_or_expr, dep_expr_or_func, indep_vars=None):
    form = characteristic_form_first_order_2vars(eq_or_expr, dep_expr_or_func, indep_vars)
    x, y = form.indep_vars
    z = sp.Symbol("s", real=True)
    u = sp.Symbol("U", real=True)
    A = sp.expand(
        form.A.subs({form.dep_function: u, x: sp.Function("X")(z), y: sp.Function("Y")(z)})
    )
    B = sp.expand(
        form.B.subs({form.dep_function: u, x: sp.Function("X")(z), y: sp.Function("Y")(z)})
    )
    C = sp.expand(
        form.C.subs({form.dep_function: u, x: sp.Function("X")(z), y: sp.Function("Y")(z)})
    )
    X = sp.Function("X")
    Y = sp.Function("Y")
    U = sp.Function("U")
    return (
        sp.Eq(
            sp.diff(X(z), z),
            A.subs(u, U(z)).subs({sp.Function("X")(z): X(z), sp.Function("Y")(z): Y(z)}),
        ),
        sp.Eq(
            sp.diff(Y(z), z),
            B.subs(u, U(z)).subs({sp.Function("X")(z): X(z), sp.Function("Y")(z): Y(z)}),
        ),
        sp.Eq(
            sp.diff(U(z), z),
            C.subs(u, U(z)).subs({sp.Function("X")(z): X(z), sp.Function("Y")(z): Y(z)}),
        ),
    )


def solve_scalar_conservation_law_riemann_burgers(u_left, u_right, *, x=None, t=None):
    x = sp.Symbol("x", real=True) if x is None else sp.sympify(x)
    t = sp.Symbol("t", positive=True, real=True) if t is None else sp.sympify(t)
    if sp.simplify(u_left - u_right) <= 0 if all(v.is_number for v in [u_left, u_right]) else False:
        return PDEIVPResult(
            "burgers_riemann_rarefaction",
            construct_burgers_rarefaction(u_left, u_right, x=x, t=t),
            {"left": u_left, "right": u_right},
        )
    s = sp.simplify((sp.sympify(u_left) + sp.sympify(u_right)) / 2)
    sol = sp.Piecewise((u_left, x < s * t), (u_right, True))
    return PDEIVPResult(
        "burgers_riemann_shock", sol, {"left": u_left, "right": u_right, "shock_speed": s}
    )


def solve_wave_equation_1d_mixed_series(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    wave_speed=1,
    length=sp.pi,
    initial_displacement=None,
    initial_velocity=None,
    left_bc="dirichlet",
    right_bc="neumann",
    terms=6,
):
    # simple mixed sine-half-integer basis
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    x, t = vars_
    c = sp.sympify(wave_speed)
    L = sp.sympify(length)
    xi = sp.Symbol("xi", real=True)
    f = _safe_sub_profile(initial_displacement if initial_displacement is not None else 0, xi)
    g = _safe_sub_profile(initial_velocity if initial_velocity is not None else 0, xi)
    series = 0
    for n in range(int(terms)):
        lam = (n + sp.Rational(1, 2)) * sp.pi / L
        basis = sp.sin(lam * xi)
        bn = 2 / L * sp.Integral(f * basis, (xi, 0, L))
        cn = 2 / (c * L * lam) * sp.Integral(g * basis, (xi, 0, L))
        series += (bn * sp.cos(c * lam * t) + cn * sp.sin(c * lam * t)) * sp.sin(lam * x)
    return PDEIVPResult(
        "wave_mixed_series",
        sp.Eq(uexpr, sp.simplify(series)),
        {
            "wave_speed": c,
            "length": L,
            "terms": int(terms),
            "left_bc": left_bc,
            "right_bc": right_bc,
        },
    )


@dataclass(frozen=True)
class DomainSpec:
    geometry: str  # e.g. whole_line, half_line, interval, rectangle, whole_plane
    variables: tuple[sp.Symbol, ...]
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitialConditionSpec:
    kind: str  # profile, displacement, velocity, cauchy_curve
    variable: sp.Symbol | None
    value: Any
    location: Any = 0


@dataclass(frozen=True)
class BoundaryConditionSpec:
    kind: str  # dirichlet, neumann, robin
    variable: sp.Symbol
    location: Any
    value: Any
    coefficient: Any | None = None


@dataclass(frozen=True)
class PDEProblemSpec:
    equation: sp.Equality
    dep_function: sp.Expr
    indep_vars: tuple[sp.Symbol, ...]
    domain: DomainSpec | None
    initial_conditions: tuple[InitialConditionSpec, ...]
    boundary_conditions: tuple[BoundaryConditionSpec, ...]
    assumptions: Any = True


@dataclass(frozen=True)
class PDESolverPlanStep:
    method: str
    reason: str
    expected_output: str
    score: int


@dataclass(frozen=True)
class PDESolverPlan:
    profile: Any
    problem: PDEProblemSpec
    steps: tuple[PDESolverPlanStep, ...]


@dataclass(frozen=True)
class PDETransformResult:
    method: str
    transformed_equation: sp.Equality | sp.Expr
    transform_variables: tuple[sp.Symbol, ...]
    inverse_representation: sp.Expr | sp.Equality | None
    details: dict


@dataclass(frozen=True)
class ReducedEquationSolveResult:
    method: str
    solved: bool
    solution: Any
    details: dict


def _as_callable_or_expr_value(obj, var):
    if callable(obj):
        return obj(var)
    return sp.sympify(obj)


def _normalize_problem_domain(indep_vars, ics=None, bcs=None, kwargs=None):
    kwargs = kwargs or {}
    if isinstance(kwargs.get("domain"), DomainSpec):
        return kwargs["domain"]
    vars_ = tuple(indep_vars)
    geom = kwargs.get("geometry")
    params = {}
    if geom is None:
        if bcs:
            bctype = bcs.get("type") if isinstance(bcs, dict) else None
            if bctype == "dirichlet_homogeneous_interval":
                geom = "interval"
                params["length"] = bcs.get("length", sp.pi)
            elif bctype == "half_line_dirichlet":
                geom = "half_line"
            elif bctype == "rectangle_dirichlet":
                geom = "rectangle"
                params["x_length"] = bcs.get("x_length", sp.pi)
                params["y_length"] = bcs.get("y_length", sp.pi)
        if geom is None:
            geom = (
                "whole_line"
                if len(vars_) == 2
                else ("whole_plane" if len(vars_) == 3 else "generic")
            )
    for k in ("length", "x_length", "y_length", "left", "right"):
        if k in kwargs:
            params[k] = kwargs[k]
    return DomainSpec(geom, vars_, params)


def _normalize_problem_conditions(indep_vars, ics=None, bcs=None):
    vars_ = tuple(indep_vars)
    x = vars_[0] if vars_ else None
    t = vars_[1] if len(vars_) > 1 else None
    y = vars_[1] if len(vars_) > 1 else None
    ic_specs = []
    bc_specs = []
    if isinstance(ics, dict):
        if "initial_profile" in ics:
            ic_specs.append(
                InitialConditionSpec(
                    "profile", x, ics["initial_profile"], ics.get("curve_value", 0)
                )
            )
        if "initial_displacement" in ics:
            ic_specs.append(
                InitialConditionSpec(
                    "displacement", x, ics["initial_displacement"], ics.get("curve_value", 0)
                )
            )
        if "initial_velocity" in ics:
            ic_specs.append(
                InitialConditionSpec(
                    "velocity", x, ics["initial_velocity"], ics.get("curve_value", 0)
                )
            )
    if isinstance(bcs, dict):
        bctype = bcs.get("type")
        if bctype == "dirichlet_homogeneous_interval":
            L = bcs.get("length", sp.pi)
            bc_specs.append(BoundaryConditionSpec("dirichlet", x, 0, 0))
            bc_specs.append(BoundaryConditionSpec("dirichlet", x, L, 0))
        elif bctype == "half_line_dirichlet":
            bc_specs.append(BoundaryConditionSpec("dirichlet", x, 0, bcs.get("value", 0)))
        elif bctype == "half_line_neumann":
            bc_specs.append(BoundaryConditionSpec("neumann", x, 0, bcs.get("value", 0)))
        elif bctype == "rectangle_dirichlet" and len(vars_) >= 2:
            Lx = bcs.get("x_length", sp.pi)
            Ly = bcs.get("y_length", sp.pi)
            bc_specs.extend(
                [
                    BoundaryConditionSpec("dirichlet", x, 0, 0),
                    BoundaryConditionSpec("dirichlet", x, Lx, 0),
                    BoundaryConditionSpec("dirichlet", t if y is None else y, 0, 0),
                    BoundaryConditionSpec("dirichlet", t if y is None else y, Ly, 0),
                ]
            )
    return tuple(ic_specs), tuple(bc_specs)


def plan_pde_solution(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None, assumptions=True, **kwargs
):
    problem, profile, candidates = preprocess_pde_problem(
        eq_or_expr,
        dep_expr_or_func,
        indep_vars,
        ics=ics,
        bcs=bcs,
        assumptions=assumptions,
        **kwargs,
    )
    steps: list[PDESolverPlanStep] = []
    # preprocessing step
    steps.append(
        PDESolverPlanStep(
            "preprocess",
            "canonicalize PDE problem and infer domain/data structure",
            "normalized problem profile",
            100,
        )
    )
    # candidate methods from ranking
    for cand in candidates:
        expected = "solution or transformed/reduced equation"
        if cand.method == "symmetry_reduction":
            expected = "reduced PDE/ODE then post-solve"
        elif "transform" in cand.method:
            expected = "transformed-domain equation and inverse representation"
        elif "separation" in cand.method:
            expected = "separated ODE family or series solution"
        steps.append(PDESolverPlanStep(cand.method, cand.reason, expected, cand.score))
    # inject data-aware methods if not already high-ranked
    meths = [s.method for s in steps]
    if (
        problem.domain
        and problem.domain.geometry in {"interval", "rectangle"}
        and "separation_of_variables" not in meths
    ):
        steps.append(
            PDESolverPlanStep(
                "separation_of_variables",
                "bounded domain favors separated eigenfunction expansions",
                "series solution / separated ODEs",
                70,
            )
        )
    if (
        problem.initial_conditions
        and any(ic.kind in {"profile", "displacement"} for ic in problem.initial_conditions)
        and problem.domain
        and problem.domain.geometry in {"whole_line", "half_line"}
    ):
        if "heat_laplace_transform" not in meths:
            steps.append(
                PDESolverPlanStep(
                    "heat_laplace_transform",
                    "time-dependent data on standard domain favors Laplace/Fourier workflow",
                    "transformed-domain ODE/PDE",
                    68,
                )
            )
    # sort by score descending but keep preprocess first
    ordered = [steps[0]] + sorted(steps[1:], key=lambda s: (-s.score, s.method))
    return PDESolverPlan(profile=profile, problem=problem, steps=tuple(ordered))


def solve_rectangle_dirichlet_laplace_series(
    dep_expr_or_func, *, x=None, y=None, x_length=sp.pi, y_length=sp.pi, boundary_top=None, terms=6
):
    """
    Solve Laplace u_xx + u_yy = 0 on 0<x<Lx, 0<y<Ly with homogeneous
    Dirichlet data on x=0,x=Lx,y=0 and prescribed top boundary u(x,Ly)=g(x)
    via a truncated sine-series.
    """
    if x is None or y is None:
        raise ValueError("x and y must be provided.")
    uexpr, _ = _dep_and_vars(dep_expr_or_func, (x, y))
    Lx = sp.sympify(x_length)
    Ly = sp.sympify(y_length)
    g = boundary_top if boundary_top is not None else sp.Function("g")
    series = 0
    coeffs = []
    for n in range(1, terms + 1):
        bn = sp.Symbol(f"b{n}")
        # If explicit profile provided, use Fourier sine coefficient; else leave symbolic.
        if isinstance(g, sp.Expr) and x in g.free_symbols:
            bn = sp.simplify(2 / Lx * sp.integrate(g * sp.sin(n * sp.pi * x / Lx), (x, 0, Lx)))
        term = (
            bn
            * sp.sin(n * sp.pi * x / Lx)
            * sp.sinh(n * sp.pi * y / Lx)
            / sp.sinh(n * sp.pi * Ly / Lx)
        )
        coeffs.append(bn)
        series += term
    return PDEIVPResult(
        method="laplace_rectangle_dirichlet_series",
        solution=sp.Eq(uexpr, sp.simplify(series)),
        details={
            "x_length": Lx,
            "y_length": Ly,
            "boundary_top": g,
            "terms": terms,
            "coefficients": tuple(coeffs),
        },
    )


def solve_heat_equation_1d_robin_series(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    diffusivity=1,
    length=sp.pi,
    initial_profile=None,
    h0=1,
    hL=1,
    terms=6,
):
    """
    Formal truncated cosine-like expansion for homogeneous Robin heat IBVP.
    For simplicity, uses a cosine basis surrogate with symbolic eigenvalues mu_n.
    """
    if x is None or t is None:
        raise ValueError("x and t must be provided.")
    uexpr, _ = _dep_and_vars(dep_expr_or_func, (x, t))
    L = sp.sympify(length)
    kappa = sp.sympify(diffusivity)
    g = initial_profile if initial_profile is not None else sp.Function("g")
    series = 0
    mus = []
    for n in range(1, terms + 1):
        mu = sp.Symbol(f"mu{n}", positive=True)
        A = sp.Symbol(f"A{n}")
        mode = sp.cos(mu * x) - sp.sympify(h0) / mu * sp.sin(mu * x)
        if isinstance(g, sp.Expr) and x in g.free_symbols:
            # symbolic weighted projection surrogate
            A = sp.simplify(sp.integrate(g * mode, (x, 0, L)))
        series += A * sp.exp(-kappa * mu**2 * t) * mode
        mus.append(mu)
    return PDEIVPResult(
        method="heat_robin_series_formal",
        solution=sp.Eq(uexpr, sp.simplify(series)),
        details={
            "length": L,
            "diffusivity": kappa,
            "h0": h0,
            "hL": hL,
            "initial_profile": g,
            "eigenvalues": tuple(mus),
            "terms": terms,
        },
    )


def solve_wave_equation_1d_laplace_sine_transform_formal(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    wave_speed=1,
    length=sp.pi,
    initial_displacement=None,
    initial_velocity=None,
    laplace_variable=None,
):
    """
    Formal Laplace-in-time + sine-series transformed form for homogeneous Dirichlet wave problems on an interval.
    """
    if x is None or t is None:
        raise ValueError("x and t must be provided.")
    uexpr, _ = _dep_and_vars(dep_expr_or_func, (x, t))
    s = laplace_variable or sp.Symbol("s", positive=True)
    n = sp.Symbol("n", integer=True, positive=True)
    c = sp.sympify(wave_speed)
    L = sp.sympify(length)
    Un = sp.Function("Un")(n, s)
    omega_n = n * sp.pi * c / L
    ode = sp.Eq((s**2 + omega_n**2) * Un, sp.Function("F_n")(n) * s + sp.Function("G_n")(n))
    inv = sp.Sum(sp.Function("u_n")(n, t) * sp.sin(n * sp.pi * x / L), (n, 1, sp.oo))
    return PDETransformResult(
        "laplace_sine_wave_formal", ode, (s,), sp.Eq(uexpr, inv), {"wave_speed": c, "length": L}
    )


def solve_burgers_ivp_characteristic_formal(
    dep_expr_or_func, *, x=None, t=None, initial_profile=None
):
    """
    Formal implicit characteristic solution of inviscid Burgers u_t + u u_x = 0.
    """
    return solve_inviscid_burgers_ivp_implicit(
        dep_expr_or_func, x=x, t=t, initial_profile=initial_profile
    )


def build_quasilinear_characteristic_odes(eq_or_expr, dep_expr_or_func, indep_vars=None):
    """
    Build characteristic ODEs for a 2D quasilinear first-order PDE
        A(x,t,u) u_x + B(x,t,u) u_t = C(x,t,u)
    as
        dx/ds = A, dt/ds = B, du/ds = C.
    """
    form = characteristic_form_first_order_2vars(eq_or_expr, dep_expr_or_func, indep_vars)
    x, t = form.indep_vars
    s = sp.Symbol("s", real=True)
    X = sp.Function("X")
    T = sp.Function("T")
    U = sp.Function("U")
    A = sp.expand(form.A.subs({x: X(s), t: T(s), form.dep_function: U(s)}))
    B = sp.expand(form.B.subs({x: X(s), t: T(s), form.dep_function: U(s)}))
    C = sp.expand(form.C.subs({x: X(s), t: T(s), form.dep_function: U(s)}))
    return (
        sp.Eq(sp.diff(X(s), s), A),
        sp.Eq(sp.diff(T(s), s), B),
        sp.Eq(sp.diff(U(s), s), C),
    )


def validate_problem_data_conditions(problem: PDEProblemSpec):
    """
    Check common IVP/IBVP data relationships before solver selection.
    Returns a dict with warnings and inferred tags.
    """
    warnings = []
    tags = set()
    geom = problem.domain.geometry if problem.domain else "generic"
    ic_kinds = {ic.kind for ic in problem.initial_conditions}
    bc_kinds = {bc.kind for bc in problem.boundary_conditions}
    if geom == "interval" and not problem.boundary_conditions:
        warnings.append("interval domain detected without boundary conditions")
    if "displacement" in ic_kinds and "velocity" not in ic_kinds:
        tags.add("wave_like_incomplete_ivp")
    if "profile" in ic_kinds:
        tags.add("has_profile_data")
    if bc_kinds == {"dirichlet"}:
        tags.add("homogeneous_dirichlet_candidate")
    if bc_kinds == {"neumann"}:
        tags.add("homogeneous_neumann_candidate")
    return {"warnings": tuple(warnings), "tags": tuple(sorted(tags))}


def canonicalize_pde_problem(eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True):
    """Return a canonical PDE object with a normalized equation and stable signature."""
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    zero = _as_zero_expr(eq_or_expr)
    expr = sp.factor_terms(sp.expand(zero))
    factor = sp.Integer(1)
    if isinstance(expr, sp.Mul):
        dep_terms = []
        for fac in expr.args:
            has_dep = uexpr in fac.free_symbols or any(
                isinstance(node, sp.Derivative) and node.expr == uexpr
                for node in sp.preorder_traversal(fac)
            )
            if has_dep:
                dep_terms.append(fac)
            else:
                factor *= fac
        if dep_terms:
            expr = sp.expand(sp.Mul(*dep_terms))
    if str(expr).startswith("-"):
        expr = sp.expand(-expr)
        factor = -factor
    signature = (tuple(str(v) for v in vars_), str(sp.expand(expr)))
    return CanonicalPDEProblem(
        equation=sp.Eq(expr, 0),
        overall_factor=sp.simplify(factor),
        signature=signature,
        details={"variables": vars_, "dependent": uexpr, "assumptions": assumptions},
    )


@dataclass(frozen=True)
class PDEProblemDomain:
    geometry: str
    spatial_interval: tuple[sp.Expr, sp.Expr] | None = None
    time_interval: tuple[sp.Expr, sp.Expr] | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class PDEProblemData:
    domain: PDEProblemDomain | None
    initial_conditions: tuple[PDEInitialCondition1D, ...]
    boundary_conditions: tuple[PDEBoundaryCondition1D, ...]
    condition_residuals: tuple[sp.Expr, ...]
    conditions_satisfied: bool


@dataclass(frozen=True)
class PDEPlanStep:
    method: str
    rationale: str
    priority: int


@dataclass(frozen=True)
class PDEPlan:
    profile: object
    data: PDEProblemData
    steps: tuple[PDEPlanStep, ...]


@dataclass(frozen=True)
class TransformProblemRepresentation:
    method: str
    transformed_equation: sp.Equality | sp.Expr
    transform_variables: tuple[sp.Symbol, ...]
    inversion_formula: sp.Expr | None
    details: dict


@dataclass(frozen=True)
class StructuredSeparationResult:
    family: str
    basis: str | None
    ansatz: sp.Equality
    separated_odes: tuple[sp.Equality, ...]
    reconstruction: sp.Expr | None
    details: dict


def _normalize_domain(domain=None, bcs=None, ics=None):
    if isinstance(domain, PDEProblemDomain):
        return domain
    if isinstance(domain, dict):
        return PDEProblemDomain(
            geometry=domain.get("geometry", domain.get("type", "unspecified")),
            spatial_interval=domain.get("spatial_interval"),
            time_interval=domain.get("time_interval"),
            metadata={
                k: v
                for k, v in domain.items()
                if k not in {"geometry", "type", "spatial_interval", "time_interval"}
            },
        )
    # infer simple domains from BC metadata
    if isinstance(bcs, dict):
        btype = bcs.get("type")
        if btype in {"dirichlet_homogeneous_interval", "neumann_homogeneous_interval"}:
            return PDEProblemDomain(
                "interval",
                spatial_interval=(0, bcs.get("length", sp.pi)),
                metadata={"bc_type": btype},
            )
    return None


def analyze_problem_data(dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None, domain=None):
    _, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    dom = _normalize_domain(domain, bcs, ics)
    norm_ics, norm_bcs = _normalize_condition_dicts(ics, bcs)
    residuals = []
    if len(vars_) >= 2:
        x = vars_[0]
        # Check boundary and initial data at shared corners when the expressions are explicit enough.
        init0 = None
        for ic in norm_ics:
            if ic.derivative_order == 0 and sp.simplify(ic.location) == 0:
                init0 = ic
                break
        if init0 is not None and callable(init0.value) is False:
            init_expr = sp.sympify(init0.value)
            for bc in norm_bcs:
                if bc.kind == "dirichlet" and callable(bc.value) is False:
                    try:
                        residuals.append(
                            sp.simplify(init_expr.subs(x, bc.location) - sp.sympify(bc.value))
                        )
                    except Exception:
                        pass
                elif bc.kind == "neumann" and callable(bc.value) is False:
                    try:
                        residuals.append(
                            sp.simplify(
                                sp.diff(init_expr, x).subs(x, bc.location) - sp.sympify(bc.value)
                            )
                        )
                    except Exception:
                        pass
    ok = all(sp.simplify(r) == 0 for r in residuals) if residuals else True
    return PDEProblemData(
        domain=dom,
        initial_conditions=norm_ics,
        boundary_conditions=norm_bcs,
        condition_residuals=tuple(residuals),
        conditions_satisfied=ok,
    )


def build_transform_representation(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, method="auto", assumptions=True, **kwargs
):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if len(vars_) != 2:
        raise NotImplementedError(
            "Transform representations are currently implemented for two independent variables."
        )
    x, t = vars_
    fam = recognize_pde_family(eq_or_expr, uexpr, vars_, assumptions=assumptions)
    if method == "auto":
        method = (
            "heat_laplace_time"
            if fam and fam.family in {"linear_second_order"}
            else "advection_fourier_space"
        )
    if method == "heat_laplace_time":
        res = solve_heat_equation_1d_laplace_transform_formal(
            uexpr,
            x=x,
            t=t,
            diffusivity=kwargs.get("diffusivity", 1),
            initial_profile=(kwargs.get("ics") or {}).get("initial_profile"),
        )
        return TransformProblemRepresentation(
            method=res.method,
            transformed_equation=res.solution,
            transform_variables=(sp.Symbol("s", positive=True, real=True),),
            inversion_formula=None,
            details=res.details,
        )
    if method == "wave_laplace_time":
        res = solve_wave_equation_1d_laplace_transform_formal(
            uexpr,
            x=x,
            t=t,
            wave_speed=kwargs.get("wave_speed", 1),
            initial_displacement=(kwargs.get("ics") or {}).get("initial_displacement"),
            initial_velocity=(kwargs.get("ics") or {}).get("initial_velocity"),
        )
        return TransformProblemRepresentation(
            method=res.method,
            transformed_equation=res.solution,
            transform_variables=(sp.Symbol("s", positive=True, real=True),),
            inversion_formula=None,
            details=res.details,
        )
    if method == "advection_fourier_space":
        res = solve_advection_equation_1d_fourier_transform(
            uexpr,
            x=x,
            t=t,
            speed=kwargs.get("speed", 1),
            reaction=kwargs.get("reaction", 0),
            initial_profile=(kwargs.get("ics") or {}).get("initial_profile", sp.Function("phi")(x)),
        )
        return TransformProblemRepresentation(
            method=res.method,
            transformed_equation=res.solution,
            transform_variables=(sp.Symbol("w", real=True),),
            inversion_formula=None,
            details=res.details,
        )
    if method == "heat_half_line_sine":
        res = solve_heat_equation_1d_half_line_transform(
            uexpr,
            x=x,
            t=t,
            diffusivity=kwargs.get("diffusivity", 1),
            initial_profile=(kwargs.get("ics") or {}).get("initial_profile"),
            boundary="dirichlet",
        )
        return TransformProblemRepresentation(
            method=res.method,
            transformed_equation=res.solution,
            transform_variables=(sp.Symbol("w", positive=True, real=True),),
            inversion_formula=None,
            details=res.details,
        )
    if method == "heat_half_line_cosine":
        res = solve_heat_equation_1d_half_line_transform(
            uexpr,
            x=x,
            t=t,
            diffusivity=kwargs.get("diffusivity", 1),
            initial_profile=(kwargs.get("ics") or {}).get("initial_profile"),
            boundary="neumann",
        )
        return TransformProblemRepresentation(
            method=res.method,
            transformed_equation=res.solution,
            transform_variables=(sp.Symbol("w", positive=True, real=True),),
            inversion_formula=None,
            details=res.details,
        )
    raise NotImplementedError(f"Unknown transform method {method!r}.")


def solve_heat_equation_1d_half_line_neumann_transform(
    dep_expr_or_func, *, x=None, t=None, diffusivity=1, initial_profile=None, fourier_variable=None
):
    return solve_heat_equation_1d_half_line_transform(
        dep_expr_or_func,
        x=x,
        t=t,
        diffusivity=diffusivity,
        initial_profile=initial_profile,
        boundary="neumann",
        fourier_variable=fourier_variable,
    )


def solve_laplace_rectangle_dirichlet_series(
    dep_expr_or_func,
    *,
    x=None,
    y=None,
    width=sp.pi,
    height=sp.pi,
    bottom=0,
    top=0,
    left=0,
    right=0,
    terms=6,
):
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, y) if x is not None and y is not None else None
    )
    x, y = vars_
    a = sp.sympify(width)
    b = sp.sympify(height)
    eta = sp.Symbol("eta", real=True)

    def _as_expr(val, var):
        if callable(val):
            return val(var)
        expr = sp.sympify(val)
        free = list(expr.free_symbols)
        return expr.subs(free[0], var) if len(free) == 1 else expr

    bottom_expr = _as_expr(bottom, x)
    top_expr = _as_expr(top, x)
    left_expr = _as_expr(left, y)
    right_expr = _as_expr(right, y)

    # Build a simple superposition with x-sine modes for bottom/top and y-sine modes for left/right.
    sol = sp.Integer(0)
    for m in range(1, int(terms) + 1):
        bm = (
            2 / a * sp.Integral(bottom_expr.subs(x, eta) * sp.sin(m * sp.pi * eta / a), (eta, 0, a))
        )
        tm = 2 / a * sp.Integral(top_expr.subs(x, eta) * sp.sin(m * sp.pi * eta / a), (eta, 0, a))
        lm = 2 / b * sp.Integral(left_expr.subs(y, eta) * sp.sin(m * sp.pi * eta / b), (eta, 0, b))
        rm = 2 / b * sp.Integral(right_expr.subs(y, eta) * sp.sin(m * sp.pi * eta / b), (eta, 0, b))
        sol += (
            bm
            * sp.sinh(m * sp.pi * (b - y) / a)
            / sp.sinh(m * sp.pi * b / a)
            * sp.sin(m * sp.pi * x / a)
        )
        sol += (
            tm * sp.sinh(m * sp.pi * y / a) / sp.sinh(m * sp.pi * b / a) * sp.sin(m * sp.pi * x / a)
        )
        sol += (
            lm
            * sp.sinh(m * sp.pi * (a - x) / b)
            / sp.sinh(m * sp.pi * a / b)
            * sp.sin(m * sp.pi * y / b)
        )
        sol += (
            rm * sp.sinh(m * sp.pi * x / b) / sp.sinh(m * sp.pi * a / b) * sp.sin(m * sp.pi * y / b)
        )
    return PDEIVPResult(
        method="laplace_rectangle_dirichlet_series",
        solution=sp.Eq(uexpr, sp.simplify(sol)),
        details={"width": a, "height": b, "terms": int(terms)},
    )


def separate_variables_with_conditions(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True, bcs=None, domain=None
):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    fam = recognize_pde_family(eq_or_expr, uexpr, vars_, assumptions=assumptions)
    sep_result = separate_variables_structured(
        eq_or_expr, uexpr, vars_, assumptions=assumptions, bcs=bcs
    )
    basis = sep_result.basis_hint
    reconstruction = None
    details = {
        "separation": sep_result,
        "family": fam,
        "domain": _normalize_domain(domain, bcs, None),
    }
    return StructuredSeparationResult(
        family=sep_result.family,
        basis=basis,
        ansatz=sep_result.ansatz,
        separated_odes=sep_result.separated_odes,
        reconstruction=reconstruction,
        details=details,
    )


def solve_reduced_equation_auto(
    reduced_eq, *, ics=None, bcs=None, assumptions=True, max_symmetry_steps=2, domain=None
):
    """
    Solve a reduced equation using the available canonical methods.

    Strategy:
      - ODE: dsolve
      - 2-variable PDE: try planned classical solve first, then pdsolve, then symmetry recursion
      - if equality with algebraic solved function, return as-is
    """
    if isinstance(reduced_eq, sp.Equality) and not any(
        isinstance(node, sp.Derivative) for node in sp.preorder_traversal(reduced_eq)
    ):
        return PDEIVPResult("algebraic_reduced_solution", reduced_eq, {"already_solved": True})

    deps = []
    for node in sp.preorder_traversal(reduced_eq):
        if isinstance(node, sp.Derivative):
            expr = node.expr
            if getattr(expr, "is_Function", False) and expr not in deps:
                deps.append(expr)
        elif getattr(node, "is_Function", False) and node not in deps:
            deps.append(node)
    if not deps:
        return PDEIVPResult("reduced_equation", reduced_eq, {"solved": False})
    dep = deps[0]
    indep = tuple(dep.args)

    derivs = [
        node
        for node in sp.preorder_traversal(reduced_eq)
        if isinstance(node, sp.Derivative) and node.expr == dep
    ]
    if derivs and all(len(d.variable_count) == 1 and len(indep) == 1 for d in derivs):
        try:
            sol = sp.dsolve(reduced_eq)
            return PDEIVPResult("dsolve_reduced_ode", sol, {"reduced_kind": "ode"})
        except Exception:
            pass

    if len(indep) == 2:
        # Reduced two-variable equations are routed back through the main dispatcher before falling back to SymPy.
        try:
            from .dispatcher import pdesolve

            return pdesolve(
                reduced_eq,
                dep,
                indep,
                ics=ics,
                bcs=bcs,
                domain=domain,
                method="auto",
                assumptions=assumptions,
                prefer_symmetry=False,
                max_symmetry_steps=max_symmetry_steps,
            )
        except Exception:
            pass
        try:
            sol = pdsolve(reduced_eq)
            return PDEIVPResult("pdsolve_reduced_pde", sol, {"reduced_kind": "pde"})
        except Exception:
            pass

    return PDEIVPResult("reduced_unsolved", reduced_eq, {"solved": False})


def _solve_via_symmetry_workflow(
    norm_eq,
    uexpr,
    vars_,
    *,
    assumptions=True,
    max_symmetry_steps=2,
    ics=None,
    bcs=None,
    domain=None,
):
    res = _reduce_and_solve_by_symmetry(
        norm_eq, uexpr, vars_, assumptions=assumptions, max_symmetry_steps=max_symmetry_steps
    )
    if isinstance(res, PDEIVPResult) and isinstance(res.solution, sp.Equality):
        return res
    # Try to interpret res as a reduction result carrying a reduced equation
    try:
        final_eq = res.solution if isinstance(res, PDEIVPResult) else res
        solved = solve_reduced_equation_auto(
            final_eq,
            ics=ics,
            bcs=bcs,
            assumptions=assumptions,
            max_symmetry_steps=max_symmetry_steps,
            domain=domain,
        )
        return PDEIVPResult(
            "symmetry_reduction_plus_postsolve",
            solved.solution,
            {"reduction_result": res, "postsolve": solved},
        )
    except Exception:
        return res


def preprocess_pde_problem(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    assumptions=True,
    max_principal_order=3,
    ics=None,
    bcs=None,
    **kwargs,
):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    zero = _as_zero_expr(eq_or_expr)
    normalized = sp.Eq(sp.expand(zero), 0)
    order = _infer_pde_order(zero, uexpr)
    first_char = None
    first_lin = None
    second_cls = None
    can = None
    conslaw = None
    solved_form = None
    details = {}
    if len(vars_) == 2 and order == 1:
        try:
            first_char = characteristic_form_first_order_2vars(normalized, uexpr, vars_)
        except Exception as exc:
            details["characteristic_data_error"] = str(exc)
        try:
            first_lin = detect_first_order_linear_form_2vars(normalized, uexpr, vars_)
        except Exception as exc:
            details["first_order_linear_error"] = str(exc)
        try:
            conslaw = detect_conservation_law_1d(normalized, uexpr, vars_)
        except Exception as exc:
            details["conservation_law_error"] = str(exc)
    if order == 2:
        try:
            second_cls = classify_linear_second_order_pde(
                normalized, uexpr, vars_, assumptions=assumptions
            )
        except Exception as exc:
            details["second_order_linear_error"] = str(exc)
    try:
        fam = recognize_pde_family(normalized, uexpr, vars_, assumptions=assumptions)
        if fam is not None:
            can = fam.family
            details["canonical_family"] = can
    except Exception as exc:
        details["canonical_family_error"] = str(exc)
    try:
        from .jet_space import (
            build_scalar_general_solved_pde_from_equation,
            build_scalar_jet_equation_from_sympy_pde,
        )

        jet, pde = build_scalar_jet_equation_from_sympy_pde(
            vars_,
            uexpr.func,
            normalized,
            max_order=max(max_principal_order, order or 1),
            dep_name=getattr(uexpr.func, "__name__", "u"),
        )
        solved_form, info = build_scalar_general_solved_pde_from_equation(
            jet, pde, max_principal_order=max_principal_order
        )
        details["principal_multiindex"] = getattr(info, "principal_multiindex", None)
    except Exception as exc:
        details["principal_solved_form_error"] = str(exc)
    return PDEProblemProfile(
        tuple(vars_),
        uexpr,
        normalized,
        zero,
        order,
        solved_form,
        first_char,
        first_lin,
        second_cls,
        can,
        conslaw,
        details,
    )


def analyze_pde_problem(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True, max_principal_order=3
):
    return preprocess_pde_problem(
        eq_or_expr,
        dep_expr_or_func,
        indep_vars,
        assumptions=assumptions,
        max_principal_order=max_principal_order,
    )


def plan_pde_solution_methods(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    assumptions=True,
    prefer_transform=False,
    prefer_separation=False,
    prefer_symmetry=False,
):
    from .classification import plan_pde_solution_methods as _plan

    return _plan(
        eq_or_expr,
        dep_expr_or_func,
        indep_vars,
        ics=ics,
        bcs=bcs,
        assumptions=assumptions,
        prefer_transform=prefer_transform,
        prefer_separation=prefer_separation,
        prefer_symmetry=prefer_symmetry,
    )


def _looks_like_plain_wave_equation(profile):
    try:
        if profile.order != 2 or len(profile.indep_vars) != 2:
            return False
        x, t = profile.indep_vars
        uexpr = profile.dep_function
        zero = sp.expand(profile.zero_expression)
        utt = sp.diff(uexpr, t, 2)
        uxx = sp.diff(uexpr, x, 2)
        A = sp.expand(sp.diff(zero, utt))
        C = sp.expand(sp.diff(zero, uxx))
        residual = sp.expand(zero - A * utt - C * uxx)
        return (
            residual == 0
            and sp.simplify(A) != 0
            and sp.simplify(C) != 0
            and A.free_symbols.isdisjoint(set(profile.indep_vars))
            and C.free_symbols.isdisjoint(set(profile.indep_vars))
        )
    except Exception:
        return False


def rank_pde_solution_methods(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    assumptions=True,
    prefer_transform=False,
    prefer_separation=False,
    prefer_symmetry=False,
):
    from .classification import rank_pde_solution_methods as _rank

    return _rank(
        eq_or_expr,
        dep_expr_or_func,
        indep_vars,
        ics=ics,
        bcs=bcs,
        assumptions=assumptions,
        prefer_transform=prefer_transform,
        prefer_separation=prefer_separation,
        prefer_symmetry=prefer_symmetry,
    )


def build_pde_solution_plan(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    domain=None,
    assumptions=True,
    prefer_transform=False,
    prefer_separation=False,
    prefer_symmetry=False,
):
    profile, candidates = rank_pde_solution_methods(
        eq_or_expr,
        dep_expr_or_func,
        indep_vars,
        ics=ics,
        bcs=bcs,
        assumptions=assumptions,
        prefer_transform=prefer_transform,
        prefer_separation=prefer_separation,
        prefer_symmetry=prefer_symmetry,
    )
    data = analyze_problem_data(dep_expr_or_func, indep_vars, ics=ics, bcs=bcs, domain=domain)
    fam = getattr(profile, "canonical_family", None)
    enhfam = None
    try:
        enhfam = recognize_pde_family(
            eq_or_expr, dep_expr_or_func, indep_vars, assumptions=assumptions
        )
    except Exception:
        pass
    steps = []
    if prefer_symmetry:
        steps.append(
            PDEPlanStep("symmetry_reduction", "caller explicitly prefers symmetry reduction", 10)
        )
    if (
        prefer_transform
        and len(
            getattr(
                profile,
                "indep_vars",
                getattr(profile, "normalized_equation", sp.Symbol("x")).free_symbols
                if hasattr(profile, "normalized_equation")
                else (),
            )
        )
        >= 0
    ):
        if ics and "initial_profile" in ics:
            if data.domain is not None and data.domain.geometry == "half_line":
                bc_kinds = {bc.kind for bc in data.boundary_conditions}
                if "neumann" in bc_kinds:
                    steps.append(
                        PDEPlanStep(
                            "heat_half_line_neumann_transform",
                            "prefer transform on half-line Neumann problem",
                            5,
                        )
                    )
                else:
                    steps.append(
                        PDEPlanStep(
                            "heat_half_line_transform",
                            "prefer transform on half-line Dirichlet problem",
                            5,
                        )
                    )
            else:
                steps.append(
                    PDEPlanStep(
                        "fourier_heat",
                        "prefer transform on whole-line heat/advection-diffusion style problem",
                        6,
                    )
                )
                steps.append(
                    PDEPlanStep(
                        "heat_laplace_transform",
                        "Laplace-in-time transform as secondary transform preference",
                        7,
                    )
                )
        if ics and ("initial_displacement" in ics or "initial_velocity" in ics):
            steps.append(
                PDEPlanStep("wave_laplace_transform", "prefer transform on wave-type IVP", 8)
            )
            steps.append(PDEPlanStep("wave_dalembert", "classical wave IVP formula", 9))
    if data.domain is not None and data.domain.geometry == "interval":
        bc_kinds = {bc.kind for bc in data.boundary_conditions}
        if bc_kinds == {"dirichlet"}:
            if any(ic.derivative_order == 1 for ic in data.initial_conditions):
                steps.append(
                    PDEPlanStep(
                        "wave_dirichlet_series",
                        "interval with Dirichlet boundaries and wave-type data suggests sine-series method",
                        22,
                    )
                )
            steps.append(
                PDEPlanStep(
                    "heat_dirichlet_series",
                    "interval with homogeneous Dirichlet boundaries suggests sine-series heat method",
                    23,
                )
            )
        elif bc_kinds == {"neumann"}:
            steps.append(
                PDEPlanStep(
                    "heat_neumann_series",
                    "interval with homogeneous Neumann boundaries suggests cosine-series heat method",
                    24,
                )
            )
    if fam in {"heat_like", "advection_diffusion_like"} or (
        enhfam and enhfam.family in {"black_scholes_like", "telegraph_like"}
    ):
        if not prefer_transform:
            steps.append(
                PDEPlanStep(
                    "fourier_heat",
                    "whole-line transform often effective for parabolic-like problems",
                    28,
                )
            )
        steps.append(
            PDEPlanStep(
                "separation_of_variables",
                "heat-like family supports condition-aware separation",
                32 if not prefer_transform else 42,
            )
        )
    elif fam == "wave_like":
        steps.append(
            PDEPlanStep(
                "separation_of_variables",
                "wave-like family supports condition-aware separation",
                34 if not prefer_transform else 44,
            )
        )
    elif fam == "laplace_helmholtz_like" or (
        enhfam and enhfam.family in {"laplace_rectangle_dirichlet_like"}
    ):
        steps.append(
            PDEPlanStep(
                "separation_of_variables",
                "elliptic family supports separated/series solutions on standard domains",
                35,
            )
        )
    for cand in candidates:
        steps.append(
            PDEPlanStep(
                cand.method,
                "; ".join(getattr(cand, "reasons", ())) or "ranked candidate",
                100 - int(getattr(cand, "score", 0)),
            )
        )
    steps.append(
        PDEPlanStep(
            "symmetry_reduction",
            "reduction remains a strong fallback and post-processing route",
            200,
        )
    )
    seen = set()
    ordered = []
    for st in sorted(steps, key=lambda s: (s.priority, s.method)):
        if st.method in seen:
            continue
        seen.add(st.method)
        ordered.append(st)
    return PDEPlan(profile=profile, data=data, steps=tuple(ordered))


def _wrap_pde_result_record(
    method_name,
    raw_result,
    plan,
    canonical_eq,
    assumptions,
    dep_expr_or_func=None,
    indep_vars=None,
    ics=None,
    bcs=None,
):
    if isinstance(raw_result, PDESolutionRecord):
        return raw_result
    actual_method = getattr(raw_result, "method", method_name)
    solution = raw_result.solution if isinstance(raw_result, PDEIVPResult) else raw_result
    details = (
        dict(getattr(raw_result, "details", {})) if isinstance(raw_result, PDEIVPResult) else {}
    )
    verification = {"verified": None}
    if isinstance(solution, sp.Equality) and dep_expr_or_func is not None:
        try:
            ver = verify_pde_solution_with_data(
                canonical_eq,
                solution,
                dep_expr_or_func,
                indep_vars,
                ics=ics,
                bcs=bcs,
                assumptions=assumptions,
            )
            verification = {
                "verified": ver.verified,
                "pde_residual": ver.pde_residual,
                "initial_residuals": ver.initial_residuals,
                "boundary_residuals": ver.boundary_residuals,
            }
        except Exception:
            pass
    return PDESolutionRecord(
        method=actual_method,
        solution=solution,
        steps=tuple(s.method for s in plan.steps),
        verification=verification,
        assumptions=assumptions,
        canonical_equation=canonical_eq,
        metadata=details,
    )


def solve_heat_equation_1d_laplace_fourier_formal(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    diffusivity=1,
    initial_profile=None,
    laplace_variable=None,
    fourier_variable=None,
):
    _, vars_ = _dep_and_vars(dep_expr_or_func, (x, t) if x is not None and t is not None else None)
    x, t = vars_
    s = (
        sp.Symbol("s", positive=True, real=True)
        if laplace_variable is None
        else sp.sympify(laplace_variable)
    )
    k = sp.Symbol("k", real=True) if fourier_variable is None else sp.sympify(fourier_variable)
    Uhat = sp.Function("Uhat")(k, s)
    phi = sp.Integer(0) if initial_profile is None else _safe_sub_profile(initial_profile, x)
    phihat = sp.fourier_transform(phi, x, k)
    alg = sp.Eq((s + sp.sympify(diffusivity) * k**2) * Uhat, phihat)
    return TransformProblemRepresentation(
        method="laplace_fourier_heat_formal",
        transformed_equation=alg,
        transform_variables=(s, k),
        inversion_formula=None,
        details={
            "transformed_unknown": Uhat,
            "initial_transform": phihat,
            "diffusivity": sp.sympify(diffusivity),
        },
    )


def solve_advection_reaction_fourier_formal(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    speed=1,
    reaction=0,
    initial_profile=None,
    fourier_variable=None,
):
    _, vars_ = _dep_and_vars(dep_expr_or_func, (x, t) if x is not None and t is not None else None)
    x, t = vars_
    k = sp.Symbol("k", real=True) if fourier_variable is None else sp.sympify(fourier_variable)
    Uhat = sp.Function("Uhat")(k, t)
    phi = sp.Integer(0) if initial_profile is None else _safe_sub_profile(initial_profile, x)
    phihat = sp.fourier_transform(phi, x, k)
    ode = sp.Eq(sp.diff(Uhat, t) + (sp.sympify(reaction) + sp.I * sp.sympify(speed) * k) * Uhat, 0)
    return TransformProblemRepresentation(
        method="fourier_advection_reaction_formal",
        transformed_equation=ode,
        transform_variables=(k,),
        inversion_formula=None,
        details={
            "transformed_unknown": Uhat,
            "initial_transform": phihat,
            "explicit_solution": sp.Eq(
                Uhat, phihat * sp.exp(-(sp.sympify(reaction) + sp.I * sp.sympify(speed) * k) * t)
            ),
        },
    )


@dataclass(frozen=True)
class QuasilinearCharacteristicODESystem2D:
    indep_vars: tuple[sp.Symbol, sp.Symbol]
    dep_function: sp.Expr
    parameter: sp.Symbol
    x_curve: sp.Expr
    t_curve: sp.Expr
    u_curve: sp.Expr
    odes: tuple[sp.Equality, sp.Equality, sp.Equality]
    details: dict


@dataclass(frozen=True)
class ConservationFormExtraction:
    indep_vars: tuple[sp.Symbol, sp.Symbol]
    dep_function: sp.Expr
    density: sp.Expr
    flux: sp.Expr
    source: sp.Expr
    normalized_equation: sp.Equality
    details: dict


def extract_conservation_form_auto(eq_or_expr, dep_expr_or_func, indep_vars=None):
    """
    Try to rewrite a first-order scalar PDE in two variables as

        rho_t + F_x = S

    Current scope:
      - linear dependence on u_t, u_x,
      - after normalization by the coefficient of u_t,
      - density rho = u when possible.

    The extracted form includes the source term,
    and it accepts constant nonzero coefficients on u_t by normalizing first.
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if len(vars_) != 2:
        raise ValueError("Expected two variables (x,t).")
    x, t = vars_
    zero = _as_zero_expr(eq_or_expr)
    ux = sp.diff(uexpr, x)
    ut = sp.diff(uexpr, t)

    # first order only
    for node in sp.preorder_traversal(zero):
        if isinstance(node, sp.Derivative) and node.expr == uexpr:
            total_order = sum(count for _, count in node.variable_count)
            if total_order > 1:
                raise NotImplementedError(
                    "Automatic conservation-form extraction currently targets first-order scalar PDEs."
                )

    try:
        poly = sp.Poly(zero, ux, ut, domain="EX")
    except Exception as exc:
        raise ValueError("Could not treat PDE as polynomial in first derivatives.") from exc
    if poly.total_degree() > 1:
        raise NotImplementedError(
            "Automatic conservation-form extraction expects linear dependence on first derivatives."
        )

    A = sp.expand(sp.diff(zero, ut))
    B = sp.expand(sp.diff(zero, ux))
    rest = sp.expand(zero.subs({ut: 0, ux: 0}))

    if sp.simplify(A) == 0:
        raise NotImplementedError("Equation is not evolutionary in u_t after linearization.")

    # Normalize by A when possible.
    if not A.free_symbols.isdisjoint({ux, ut}):
        raise NotImplementedError("Coefficient of u_t depends on derivatives.")

    Bn = sp.expand(B / A)
    Rn = sp.expand(rest / A)

    usym = sp.Symbol("_ucons_auto")
    B_hold = sp.expand(Bn.subs({uexpr: usym}))
    R_hold = sp.expand(Rn.subs({uexpr: usym}))

    # Construct a flux whose u-derivative matches Bn.
    F_u = sp.integrate(B_hold, usym)
    Fx_from_u = sp.diff(F_u, x)
    correction = sp.simplify(R_hold - Fx_from_u)

    # If correction depends on u, keep it as a source instead of failing.
    if correction.has(usym):
        F_hold = sp.expand(F_u)
        source_hold = sp.expand(-correction)
    else:
        xcorr = sp.integrate(correction, x)
        F_hold = sp.expand(F_u + xcorr)
        source_hold = sp.Integer(0)

    F_expr = sp.expand(F_hold.subs({usym: uexpr}))
    S_expr = sp.expand(source_hold.subs({usym: uexpr}))

    # Verify the reconstruction.
    reconstructed = sp.expand(
        ut
        + sp.diff(F_hold, x).subs({usym: uexpr})
        + sp.diff(F_hold, usym).subs({usym: uexpr}) * ux
        - S_expr
    )
    if sp.simplify(reconstructed - (ut + Bn * ux + Rn)) != 0:
        raise NotImplementedError("Could not verify automatically extracted conservation form.")

    return ConservationFormExtraction(
        indep_vars=(x, t),
        dep_function=uexpr,
        density=uexpr,
        flux=F_expr,
        source=S_expr,
        normalized_equation=sp.Eq(ut + sp.diff(F_expr, x) + sp.diff(F_expr, uexpr) * ux, S_expr),
        details={"A": A, "B_normalized": Bn, "rest_normalized": Rn},
    )


def build_quasilinear_characteristic_system_2vars_robust(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    initial_curve=None,
    initial_data=None,
    parameter=None,
):
    """
    Build a parametric characteristic ODE system for a two-variable first-order
    quasilinear PDE

        A(x,t,u) u_x + B(x,t,u) u_t = C(x,t,u).

    If initial_curve and initial_data are supplied, include parametric initial
    conditions:
        x(0,xi), t(0,xi), u(0,xi).
    """
    form = characteristic_form_first_order_2vars(eq_or_expr, dep_expr_or_func, indep_vars)
    x, t = form.indep_vars
    s = sp.Symbol("s", real=True) if parameter is None else sp.sympify(parameter)
    xi = sp.Symbol("xi", real=True)
    X = sp.Function("X")
    T = sp.Function("T")
    U = sp.Function("U")

    Xs = X(s)
    Ts = T(s)
    Us = U(s)

    repl = {x: Xs, t: Ts, form.dep_function: Us}
    A = sp.expand(form.A.subs(repl))
    B = sp.expand(form.B.subs(repl))
    C = sp.expand(form.C.subs(repl))

    x0 = _safe_sub_profile_general(initial_curve[0], xi) if initial_curve is not None else xi
    t0 = (
        _safe_sub_profile_general(initial_curve[1], xi)
        if initial_curve is not None
        else sp.Integer(0)
    )
    u0 = (
        _safe_sub_profile_general(initial_data, xi)
        if initial_data is not None
        else sp.Symbol("u0")(xi)
    )

    return QuasilinearCharacteristicODESystem2D(
        indep_vars=(x, t),
        dep_function=form.dep_function,
        parameter=s,
        x_curve=x0,
        t_curve=t0,
        u_curve=u0,
        odes=(
            sp.Eq(sp.diff(Xs, s), A),
            sp.Eq(sp.diff(Ts, s), B),
            sp.Eq(sp.diff(Us, s), C),
        ),
        details={"characteristic_form": form, "initial_parameter": xi},
    )


def solve_viscous_burgers_cole_hopf_formal(
    dep_expr_or_func, *, x=None, t=None, viscosity=1, initial_profile=None
):
    """
    Formal Cole-Hopf representation for
        u_t + u u_x = nu u_xx
    on the line.
    """
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    x, t = vars_
    nu = sp.sympify(viscosity)
    xi = sp.Symbol("xi", real=True)
    if initial_profile is None:
        raise ValueError("initial_profile is required for formal Cole-Hopf solution.")
    g = _safe_sub_profile_general(initial_profile, xi)
    phi0 = sp.exp(-sp.Integral(g, (xi, 0, x)) / (2 * nu))
    eta = sp.Symbol("eta", real=True)
    heat_kernel = sp.exp(-((x - eta) ** 2) / (4 * nu * t)) / sp.sqrt(4 * sp.pi * nu * t)
    phi = sp.Integral(heat_kernel * phi0.subs(x, eta), (eta, -sp.oo, sp.oo))
    usol = sp.simplify(-2 * nu * sp.diff(sp.log(phi), x))
    return PDEIVPResult(
        method="viscous_burgers_cole_hopf_formal",
        solution=sp.Eq(uexpr, usol),
        details={"viscosity": nu, "phi_initial": phi0, "heat_potential": phi},
    )


def entropy_select_riemann_branch_scalar(flux, u_left, u_right, *, u_symbol=None, assumptions=True):
    """
    Entropy-aware branch selection for scalar conservation laws with flux f(u).

    Returns a dict describing whether the entropy solution is a shock or a
    rarefaction, using convexity/concavity heuristics.
    """
    u = sp.Symbol("u", real=True) if u_symbol is None else sp.sympify(u_symbol)
    f = sp.sympify(flux)
    fpp = sp.simplify(sp.diff(f, u, 2))
    ul = sp.sympify(u_left)
    ur = sp.sympify(u_right)

    def _sign(v):
        try:
            return sp.sign(sp.simplify(v))
        except Exception:
            return sp.S.NaN

    convexity = _sign(fpp)
    diff_sign = _sign(ur - ul)

    if convexity == 1:
        branch = "rarefaction" if diff_sign == 1 else "shock"
    elif convexity == -1:
        branch = "shock" if diff_sign == 1 else "rarefaction"
    else:
        # fallback to Burgers/Lax-style default when ordering is known
        branch = "rarefaction" if diff_sign == 1 else "shock"

    return {
        "branch": branch,
        "flux": f,
        "u_symbol": u,
        "convexity_sign": convexity,
        "ordering_sign": diff_sign,
        "shock_speed": rankine_hugoniot_speed(f, ul, ur, u_symbol=u) if branch == "shock" else None,
        "left_speed": sp.simplify(sp.diff(f, u).subs(u, ul)),
        "right_speed": sp.simplify(sp.diff(f, u).subs(u, ur)),
    }


def solve_burgers_family(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None, assumptions=True
):
    """
    Solve supported Burgers-family equations using structure-aware routing.

    Supports:
      - inviscid Burgers IVP by implicit characteristics,
      - viscous Burgers formal Cole-Hopf solution,
      - Burgers-family recognition with optional forcing metadata.
    """
    recog = detect_burgers_family(eq_or_expr, dep_expr_or_func, indep_vars)
    if recog is None:
        raise NotImplementedError("Equation does not match a supported Burgers family.")

    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    x = vars_[0]
    t = vars_[1]
    ics = {} if ics is None else dict(ics)

    if recog.family == "inviscid_burgers":
        if "initial_profile" in ics:
            return solve_inviscid_burgers_ivp_implicit(
                uexpr, x=x, t=t, initial_profile=ics["initial_profile"]
            )
        return PDEIVPResult(
            "burgers_family_recognition", recog.normalized_equation, {"family": recog}
        )

    if recog.family == "viscous_burgers":
        if (
            "initial_profile" in ics
            and sp.simplify(recog.details.get("forcing", 0)) == 0
            and sp.simplify(recog.details.get("alpha", 1) - 1) == 0
        ):
            return solve_viscous_burgers_cole_hopf_formal(
                uexpr,
                x=x,
                t=t,
                viscosity=recog.details["nu"],
                initial_profile=ics["initial_profile"],
            )
        return PDEIVPResult(
            "burgers_family_recognition", recog.normalized_equation, {"family": recog}
        )

    return PDEIVPResult("burgers_family_recognition", recog.normalized_equation, {"family": recog})


def solve_scalar_conservation_law_riemann_general(
    flux, u_left, u_right, *, x=None, t=None, u_symbol=None, entropy="lax"
):
    x = sp.Symbol("x", real=True) if x is None else sp.sympify(x)
    t = sp.Symbol("t", positive=True, real=True) if t is None else sp.sympify(t)
    u = sp.Symbol("u", real=True) if u_symbol is None else sp.sympify(u_symbol)
    f = sp.sympify(flux)

    sel = entropy_select_riemann_branch_scalar(f, u_left, u_right, u_symbol=u)
    ul = sp.sympify(u_left)
    ur = sp.sympify(u_right)
    xi = sp.simplify(x / t)
    fp = sp.diff(f, u)

    if sel["branch"] == "shock":
        s = sel["shock_speed"]
        sol = sp.Piecewise((ul, x < s * t), (ur, True))
        return PDEIVPResult("scalar_conservation_riemann_shock", sol, {"selection": sel})

    try:
        inv_candidates = sp.solve(sp.Eq(fp, xi), u, dict=False)
    except Exception:
        inv_candidates = []
    if not isinstance(inv_candidates, (list, tuple)):
        inv_candidates = [inv_candidates]

    chosen = None
    for cand in inv_candidates:
        if cand is None:
            continue
        if cand.has(xi) or cand.has(x) or cand.has(t):
            chosen = sp.simplify(cand)
            break
    if chosen is None:
        chosen = sp.Function("finv")(xi)

    left_speed = sel["left_speed"]
    right_speed = sel["right_speed"]
    sol = sp.Piecewise(
        (ul, xi <= left_speed), (chosen, sp.And(xi >= left_speed, xi <= right_speed)), (ur, True)
    )
    return PDEIVPResult("scalar_conservation_riemann_rarefaction", sol, {"selection": sel})


def solve_quasilinear_pde_characteristics_implicit(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, initial_profile=None, initial_curve_value=0
):
    form = characteristic_form_first_order_2vars(eq_or_expr, dep_expr_or_func, indep_vars)
    x, t = form.indep_vars
    uexpr = form.dep_function

    if initial_profile is None:
        raise ValueError("initial_profile is required for implicit characteristic solving.")

    nonautonomous = False
    for coeff in (form.A, form.B, form.C):
        coeff_sub = sp.expand(coeff.xreplace({uexpr: sp.Symbol("_uauto")}))
        if not coeff_sub.free_symbols.isdisjoint({x, t}):
            nonautonomous = True
            break

    B = sp.expand(form.B)
    if nonautonomous:
        s = sp.Symbol("s", real=True)
        xi = sp.Symbol("xi", real=True)
        X = sp.Function("X")
        T = sp.Function("T")
        U = sp.Function("U")
        a_form = sp.simplify(form.A.subs({x: X(s), t: T(s), uexpr: U(s)}))
        b_form = sp.simplify(form.B.subs({x: X(s), t: T(s), uexpr: U(s)}))
        c_form = sp.simplify(form.C.subs({x: X(s), t: T(s), uexpr: U(s)}))
        gxi = _safe_sub_profile_general(initial_profile, xi)
        odes = (
            sp.Eq(sp.diff(X(s), s), a_form),
            sp.Eq(sp.diff(T(s), s), b_form),
            sp.Eq(sp.diff(U(s), s), -c_form),
        )
        ics = (
            sp.Eq(X(0), xi),
            sp.Eq(T(0), initial_curve_value),
            sp.Eq(U(0), gxi),
        )
        solved_relations = []
        explicit = False
        shock_meta = {"crossing_detected": None, "analysis": "formal_only"}
        try:
            sx = sp.dsolve(odes[0], ics=ics[0])
            st = sp.dsolve(odes[1], ics=ics[1])
            su = None
            if not sp.sympify(c_form).has(U(s)):
                su = sp.dsolve(odes[2], ics=ics[2])
            elif sp.simplify(c_form) == 0:
                su = sp.Eq(U(s), gxi)
            solved_relations = [sx, st] + ([su] if su is not None else [])
            explicit = True
            try:
                # crude shock-crossing diagnostic from x_xi when possible
                Xrhs = sx.rhs if isinstance(sx, sp.Equality) else None
                if Xrhs is not None:
                    dxi = sp.diff(Xrhs, xi)
                    shock_meta = {
                        "crossing_indicator": dxi,
                        "crossing_detected": sp.simplify(dxi == 0),
                        "analysis": "explicit_characteristics",
                    }
            except Exception:
                pass
        except Exception:
            solved_relations = list(odes) + list(ics)
        details = {
            "speed_function": form.A,
            "time_flow_function": form.B,
            "source_function": form.C,
            "initial_curve_value": initial_curve_value,
            "implicit": True,
            "formal_characteristic_system": not explicit,
            "explicit_characteristic_system": explicit,
            "footpoint_symbol": xi,
            "shock_diagnostics": shock_meta,
        }
        return PDEIVPResult(
            method="quasilinear_implicit_characteristics",
            solution=tuple(solved_relations),
            details=details,
        )
    if sp.simplify(B) == 0:
        raise NotImplementedError(
            "Need nonzero coefficient on u_t for implicit characteristic solving."
        )

    xi = sp.Symbol("xi", real=True)
    gxi = _safe_sub_profile_general(initial_profile, xi)
    tau = sp.expand(t - initial_curve_value)

    au = sp.simplify(form.A / B)
    cu = sp.simplify(form.C / B)

    uaux = sp.Symbol("Uchar", real=True)
    au_sym = sp.expand(au.xreplace({uexpr: uaux}))
    cu_sym = sp.expand(cu.xreplace({uexpr: uaux}))

    if sp.simplify(cu_sym) == 0:
        sol = sp.Eq(uexpr, sp.expand(gxi.subs(xi, x - au.subs({uexpr: uexpr}) * tau)))
        return PDEIVPResult(
            method="quasilinear_implicit_characteristics",
            solution=sol,
            details={
                "speed_function": au,
                "source_function": cu,
                "initial_curve_value": initial_curve_value,
                "implicit": True,
                "shock_diagnostics": {
                    "crossing_indicator": sp.diff(x - au_sym * tau, uaux),
                    "analysis": "autonomous_implicit",
                },
            },
        )

    Iu = sp.Integral(1 / cu_sym, (uaux, gxi, uexpr))
    Ix = sp.Integral(au_sym / cu_sym, (uaux, gxi, uexpr))
    relations = (sp.Eq(tau, Iu), sp.Eq(x - xi, Ix))
    return PDEIVPResult(
        method="quasilinear_implicit_characteristics",
        solution=relations,
        details={
            "speed_function": au,
            "source_function": cu,
            "initial_curve_value": initial_curve_value,
            "implicit": True,
            "shock_diagnostics": {
                "crossing_indicator": sp.diff(Ix, xi),
                "analysis": "autonomous_integral_relations",
            },
        },
    )


def fit_constant_coefficient_solution_2d(
    general_solution_result, uexpr, vars_, *, ics=None, bcs=None
):
    if ics is None:
        return None
    x, t = vars_
    fsols = general_solution_result.details.get("factor_solutions", [])
    p = sp.expand(general_solution_result.details.get("particular", 0))

    # single transport family with profile data u(x,t0)=f(x)
    if len(fsols) == 1 and "initial_profile" in ics:
        fs = fsols[0]
        z0 = sp.expand(fs.invariant.subs(t, sp.sympify(ics.get("curve_value", 0))))
        if sp.simplify(sp.diff(z0, x)) != 0:
            f = _safe_sub_profile_general(ics["initial_profile"], x)
            F = fs.arbitrary_functions[0]
            base = sp.expand(f - p.subs(t, sp.sympify(ics.get("curve_value", 0))))
            try:
                inv = sp.solve(sp.Eq(sp.Symbol("zeta"), z0), x, dict=False)
            except Exception:
                inv = []
            if inv:
                zeta = sp.Symbol("zeta", real=True)
                Fexpr = sp.expand(base.subs(x, inv[0].subs(sp.Symbol("zeta"), zeta)))
                full = sp.expand(
                    general_solution_result.solution.rhs.subs(
                        F(fs.invariant), Fexpr.subs(zeta, fs.invariant)
                    )
                )
                return PDEGeneralSolutionResult(
                    "linear_constant_coefficient_factored_fitted_single_family",
                    sp.Eq(uexpr, full),
                    {"base": general_solution_result},
                )
    return None


def solve_linear_constant_coefficient_pde_bvp_2d(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None, assumptions=True
):
    """Canonical-first restricted BVP/IVP fitting wrapper for 2D constant-coefficient PDEs."""
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    base = solve_linear_constant_coefficient_pde(
        eq_or_expr, uexpr, vars_, ics=None, bcs=None, assumptions=assumptions, canonical_first=True
    )
    if isinstance(base, PDEGeneralSolutionResult):
        fitted = fit_constant_coefficient_solution_2d(base, uexpr, vars_, ics=ics, bcs=bcs)
        if fitted is not None:
            return fitted
    return base


def solve_euler_bernoulli_beam_freefree_ibvp(
    dep_expr_or_func,
    *,
    x,
    t,
    length,
    stiffness=1,
    initial_displacement=None,
    initial_velocity=None,
    n_terms=8,
):
    """
    Restricted spectral solver for a free-free Euler-Bernoulli beam on [0,L].
    Uses cosine basis as a starter formal model.
    """
    uexpr, _ = _dep_and_vars(dep_expr_or_func, (x, t))
    L = sp.sympify(length)
    c = sp.sympify(stiffness)
    if initial_displacement is None or initial_velocity is None:
        raise ValueError("Both initial_displacement and initial_velocity are required.")
    f = _safe_sub_profile_general(initial_displacement, x)
    g = _safe_sub_profile_general(initial_velocity, x)
    n = sp.Symbol("n", integer=True, nonnegative=True)
    omega = sp.sqrt(c) * (n * sp.pi / L) ** 2
    # include n=0 rigid mode formally
    an = sp.Piecewise(
        (sp.integrate(f, (x, 0, L)) / sp.sqrt(L), sp.Eq(n, 0)),
        (sp.sqrt(2 / L) * sp.integrate(f * sp.cos(n * sp.pi * x / L), (x, 0, L)), True),
    )
    series = an.subs(n, 0) + sp.Sum(
        sp.sqrt(2 / L)
        * sp.cos(n * sp.pi * x / L)
        * (
            (sp.sqrt(2 / L) * sp.integrate(f * sp.cos(n * sp.pi * x / L), (x, 0, L)))
            * sp.cos(omega * t)
            + (sp.sqrt(2 / L) * sp.integrate(g * sp.cos(n * sp.pi * x / L), (x, 0, L)))
            * sp.sin(omega * t)
            / omega
        ),
        (n, 1, n_terms),
    )
    return SpectralPDEResult(
        "free_free_beam_spectral",
        sp.Eq(uexpr, series),
        {"length": L, "stiffness": c, "basis": "cosine"},
    )
