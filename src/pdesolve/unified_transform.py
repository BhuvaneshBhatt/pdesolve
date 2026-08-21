from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import sympy as sp

from .results import SolverMethodResult


@dataclass
class EvolutionPDEProfile:
    """Recognized constant-coefficient scalar evolution PDE profile.

    The normalized PDE is interpreted as

        a_t * q_t + sum_j coeffs[j] * d^j q / dx^j = 0,

    with constant coefficients.  The dispersion relation is defined by the
    Fourier ansatz ``exp(I*k*x - omega*t)``.
    """

    family_name: str
    order_x: int
    time_coefficient: sp.Expr
    coeffs: dict[int, sp.Expr]
    dispersion_relation: sp.Expr
    notes: str = ""


@dataclass(frozen=True, kw_only=True)
class UnifiedTransformResult(SolverMethodResult):
    dispersion_relation: sp.Expr
    domain: str
    is_formal: bool = False
    notes: str = ""
    profile: Optional[EvolutionPDEProfile] = None


def _as_expr(eq: sp.Equality | sp.Expr) -> sp.Expr:
    return (
        sp.simplify(eq.lhs - eq.rhs) if isinstance(eq, sp.Equality) else sp.simplify(eq)
    )


def _parse_initial_condition(
    initial_condition: sp.Equality, u: sp.Function, x: sp.Symbol, t: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    """Parse data prescribed on a constant-time slice ``t = t0``.

    Returns ``(t0, profile_expr)`` where ``profile_expr`` is expressed as a
    function of ``x``. Both argument orders ``u(x, t0)`` and ``u(t0, x)`` are
    accepted as an alternate input form.
    """
    lhs = initial_condition.lhs
    rhs = initial_condition.rhs
    if getattr(lhs, "func", None) != u or len(getattr(lhs, "args", ())) != 2:
        raise NotImplementedError(
            "Expected an initial condition for the dependent variable on a constant-time slice"
        )
    a0, a1 = lhs.args
    if a0 == x and not sp.sympify(a1).has(x, t):
        return sp.sympify(a1), rhs
    if a1 == x and not sp.sympify(a0).has(x, t):
        return sp.sympify(a0), rhs
    raise NotImplementedError(
        "Expected an initial condition of the form u(x, t0) == q0(x) on a constant-time slice"
    )


def _parse_boundary_value(
    boundary_conditions: Sequence[sp.Equality], target: sp.Expr
) -> Optional[sp.Expr]:
    for bc in boundary_conditions:
        if bc.lhs == target:
            return bc.rhs
    return None


def _extract_constant_coeff_evolution(
    eq: sp.Equality | sp.Expr, u: sp.Function, x: sp.Symbol, t: sp.Symbol
) -> Optional[tuple[sp.Expr, dict[int, sp.Expr]]]:
    """Parse ``a_t q_t + sum_j a_j q_{x^j} = 0`` with constant coefficients."""
    expr = sp.expand(_as_expr(eq))
    q = u(x, t)
    qt = sp.diff(q, t)

    # Reject mixed derivatives and higher t-derivatives.
    for deriv in expr.atoms(sp.Derivative):
        counts = deriv.variable_count
        tx = {var: mult for var, mult in counts}
        if tx.get(t, 0) > 1:
            return None
        if tx.get(t, 0) >= 1 and tx.get(x, 0) >= 1:
            return None
        if any(v not in {x, t} for v, _ in counts):
            return None

    a_t = sp.simplify(expr.coeff(qt))
    if a_t == 0:
        return None
    expr = sp.simplify(expr - a_t * qt)

    coeffs: dict[int, sp.Expr] = {}
    max_order = 0
    for deriv in expr.atoms(sp.Derivative):
        counts = dict(deriv.variable_count)
        max_order = max(max_order, counts.get(x, 0))

    for j in range(max_order, 0, -1):
        d = sp.diff(q, x, j)
        c = sp.simplify(expr.coeff(d))
        if c != 0:
            coeffs[j] = c
            expr = sp.simplify(expr - c * d)

    c0 = sp.simplify(expr.coeff(q))
    if c0 != 0:
        coeffs[0] = c0
        expr = sp.simplify(expr - c0 * q)

    if sp.simplify(expr) != 0:
        return None

    for c in [a_t, *coeffs.values()]:
        if not c.free_symbols.isdisjoint({x, t}):
            return None
    return a_t, coeffs


def determine_dispersion_relation(
    eq: sp.Equality | sp.Expr, u: sp.Function, vars: tuple[sp.Symbol, sp.Symbol]
) -> sp.Expr:
    x, t = vars
    parsed = _extract_constant_coeff_evolution(eq, u, x, t)
    if parsed is None:
        raise NotImplementedError(
            "Could not parse a scalar constant-coefficient evolution equation"
        )
    a_t, coeffs = parsed
    k = sp.symbols("k", complex=True)
    omega = sp.simplify(sum(c * (sp.I * k) ** j for j, c in coeffs.items()) / a_t)
    return sp.simplify(omega)


def recognize_evolution_pde(
    eq: sp.Equality | sp.Expr, u: sp.Function, vars: tuple[sp.Symbol, sp.Symbol]
) -> Optional[EvolutionPDEProfile]:
    """Classify supported scalar constant-coefficient evolution PDE families."""
    x, t = vars
    parsed = _extract_constant_coeff_evolution(eq, u, x, t)
    if parsed is None:
        return None
    a_t, coeffs = parsed
    omega = determine_dispersion_relation(eq, u, vars)
    nonzero_orders = sorted(j for j, c in coeffs.items() if c != 0)
    order_x = max(nonzero_orders) if nonzero_orders else 0

    family = "constant_coefficient_evolution"
    notes = "Generic scalar constant-coefficient evolution equation."

    if set(nonzero_orders).issubset({0, 1}):
        family = "advection_like"
        notes = "First-order transport/advection-reaction family."
    elif set(nonzero_orders).issubset({2}):
        c2 = sp.simplify(coeffs.get(2, 0))
        a_re, a_im = [
            sp.simplify(part) for part in sp.expand_complex(a_t).as_real_imag()
        ]
        _, c2_im = [sp.simplify(part) for part in sp.expand_complex(c2).as_real_imag()]
        if a_re == 0 and a_im != 0 and c2_im == 0:
            family = "schrodinger_like"
            notes = "Second-order dispersive Schrödinger-type family."
        else:
            family = "heat_like"
            notes = "Second-order parabolic heat/advection-diffusion family."
    elif set(nonzero_orders).issubset({3}):
        family = "airy_like"
        notes = "Third-order dispersive Airy/linearized-KdV-type family."

    return EvolutionPDEProfile(
        family_name=family,
        order_x=order_x,
        time_coefficient=a_t,
        coeffs=coeffs,
        dispersion_relation=omega,
        notes=notes,
    )


def solve_unified_transform_whole_line(
    eq: sp.Equality | sp.Expr,
    initial_condition: sp.Equality,
    u: sp.Function,
    vars: tuple[sp.Symbol, sp.Symbol],
) -> UnifiedTransformResult:
    """Return the whole-line Fourier / unified-transform representation."""
    x, t = vars
    t0, q0 = _parse_initial_condition(initial_condition, u, x, t)
    profile = recognize_evolution_pde(eq, u, vars)
    if profile is None:
        raise NotImplementedError(
            "Could not recognize a scalar constant-coefficient evolution PDE"
        )

    k = sp.symbols("k", complex=True)
    q0hat = sp.FourierTransform(q0, x, k)
    time_shift = sp.simplify(t - t0)
    solution = sp.Integral(
        q0hat
        * sp.exp(sp.I * k * x - profile.dispersion_relation * time_shift)
        / (2 * sp.pi),
        (k, -sp.oo, sp.oo),
    )
    return UnifiedTransformResult(
        solution=solution,
        dispersion_relation=profile.dispersion_relation,
        domain="whole_line",
        method_family="unified_transform_whole_line",
        is_formal=False,
        notes=f"Whole-line spectral representation via Fourier/unified transform for {profile.family_name}.",
        profile=profile,
    )


def _solve_half_line_advection(
    profile: EvolutionPDEProfile,
    initial_condition: sp.Equality,
    boundary_conditions: Sequence[sp.Equality],
    u: sp.Function,
    vars: tuple[sp.Symbol, sp.Symbol],
) -> Optional[UnifiedTransformResult]:
    x, t = vars
    a_t = profile.time_coefficient
    coeffs = profile.coeffs
    if profile.family_name != "advection_like":
        return None

    t0, q0 = _parse_initial_condition(initial_condition, u, x, t)
    g0 = _parse_boundary_value(boundary_conditions, u(0, t))
    if g0 is None:
        return None
    a = sp.simplify(coeffs.get(1, 0) / a_t)
    b = sp.simplify(coeffs.get(0, 0) / a_t)
    if a.is_number and a.is_real and a <= 0:
        return None

    time_shift = sp.simplify(t - t0)
    initial_branch = sp.exp(-b * time_shift) * q0.subs(x, x - a * time_shift)
    boundary_branch = sp.exp(-b * x / a) * g0.subs(t, t - x / a)
    solution = sp.Piecewise(
        (initial_branch, sp.Ge(x - a * time_shift, 0)), (boundary_branch, True)
    )
    return UnifiedTransformResult(
        solution=solution,
        dispersion_relation=profile.dispersion_relation,
        domain="half_line",
        method_family="unified_transform_half_line_advection",
        is_formal=False,
        notes="Explicit half-line inflow solution obtained from characteristics.",
        profile=profile,
    )


def _solve_half_line_heat_dirichlet(
    profile: EvolutionPDEProfile,
    initial_condition: sp.Equality,
    boundary_conditions: Sequence[sp.Equality],
    u: sp.Function,
    vars: tuple[sp.Symbol, sp.Symbol],
) -> Optional[UnifiedTransformResult]:
    """Dirichlet half-line heat solver in integral form."""
    x, t = vars
    if profile.family_name != "heat_like":
        return None
    a_t = profile.time_coefficient
    coeffs = profile.coeffs
    if set(coeffs) - {2}:
        return None
    kappa = sp.simplify(-coeffs.get(2, 0) / a_t)
    if not kappa.free_symbols.isdisjoint({x, t}):
        return None

    t0, q0 = _parse_initial_condition(initial_condition, u, x, t)
    g0 = _parse_boundary_value(boundary_conditions, u(0, t))
    if g0 is None:
        return None

    xi = sp.symbols("xi", positive=True, real=True)
    tau = sp.symbols("tau", real=True)
    time_shift = sp.simplify(t - t0)
    G1 = sp.exp(-((x - xi) ** 2) / (4 * kappa * time_shift)) / sp.sqrt(
        4 * sp.pi * kappa * time_shift
    )
    G2 = sp.exp(-((x + xi) ** 2) / (4 * kappa * time_shift)) / sp.sqrt(
        4 * sp.pi * kappa * time_shift
    )
    initial_part = sp.Integral((G1 - G2) * q0.subs(x, xi), (xi, 0, sp.oo))

    boundary_kernel = (
        x
        * sp.exp(-(x**2) / (4 * kappa * (t - tau)))
        / (2 * sp.sqrt(sp.pi * kappa) * (t - tau) ** sp.Rational(3, 2))
    )
    boundary_part = sp.Integral(boundary_kernel * g0.subs(t, tau), (tau, t0, t))

    return UnifiedTransformResult(
        solution=initial_part + boundary_part,
        dispersion_relation=profile.dispersion_relation,
        domain="half_line",
        method_family="unified_transform_half_line_heat_dirichlet",
        is_formal=False,
        notes="Classical half-line heat integral, equivalent to the unified-transform representation.",
        profile=profile,
    )


def _solve_half_line_schrodinger_dirichlet_zero(
    profile: EvolutionPDEProfile,
    initial_condition: sp.Equality,
    boundary_conditions: Sequence[sp.Equality],
    u: sp.Function,
    vars: tuple[sp.Symbol, sp.Symbol],
) -> Optional[UnifiedTransformResult]:
    """Half-line Schrödinger with homogeneous Dirichlet data via boundary elimination.

    This is the first genuinely nontrivial unified-transform branch beyond the
    minimal transport/heat cases.  For

        I*q_t + kappa*q_xx = 0,  x > 0, t > 0,
        q(x,0) = q0(x),        q(0,t) = 0,

    the global relation can be eliminated and the result reduces to the odd-
    extension method image formula, which is equivalent to the Fokas
    representation.
    """
    x, t = vars
    if profile.family_name != "schrodinger_like":
        return None
    if set(profile.coeffs) - {2}:
        return None

    g0 = _parse_boundary_value(boundary_conditions, u(0, t))
    if g0 is None or sp.simplify(g0) != 0:
        return None

    a_t = profile.time_coefficient
    coeffs = profile.coeffs
    kappa = sp.simplify(coeffs.get(2, 0) / a_t / sp.I)
    if kappa == 0 or not kappa.free_symbols.isdisjoint({x, t}):
        return None

    t0, q0 = _parse_initial_condition(initial_condition, u, x, t)
    xi = sp.symbols("xi", positive=True, real=True)
    time_shift = sp.simplify(t - t0)
    kernel_minus = sp.exp(sp.I * (x - xi) ** 2 / (4 * kappa * time_shift)) / sp.sqrt(
        4 * sp.pi * sp.I * kappa * time_shift
    )
    kernel_plus = sp.exp(sp.I * (x + xi) ** 2 / (4 * kappa * time_shift)) / sp.sqrt(
        4 * sp.pi * sp.I * kappa * time_shift
    )
    solution = sp.Integral(
        (kernel_minus - kernel_plus) * q0.subs(x, xi), (xi, 0, sp.oo)
    )
    return UnifiedTransformResult(
        solution=solution,
        dispersion_relation=profile.dispersion_relation,
        domain="half_line",
        method_family="unified_transform_half_line_schrodinger_dirichlet_zero",
        is_formal=False,
        notes="Half-line Schrödinger solution with homogeneous Dirichlet data after eliminating the unknown boundary transform (equivalent to the odd-extension/Fokas formula).",
        profile=profile,
    )


def _formal_half_line_representation(
    profile: EvolutionPDEProfile,
    initial_condition: sp.Equality,
    boundary_conditions: Sequence[sp.Equality],
    u: sp.Function,
    vars: tuple[sp.Symbol, sp.Symbol],
) -> UnifiedTransformResult:
    """Return a minimal formal spectral representation on the half-line."""
    x, t = vars
    t0, q0 = _parse_initial_condition(initial_condition, u, x, t)
    omega = profile.dispersion_relation
    k = sp.symbols("k", complex=True)
    s = sp.symbols("s", real=True, nonnegative=True)

    q0hat = sp.Integral(q0.subs(x, s) * sp.exp(-sp.I * k * s), (s, 0, sp.oo))
    boundary_symbol = sp.Function("G")(k, t)
    solution = sp.Integral(
        q0hat * sp.exp(sp.I * k * x - omega * (t - t0)) / (2 * sp.pi),
        (k, -sp.oo, sp.oo),
    ) + sp.Integral(
        boundary_symbol * sp.exp(sp.I * k * x - omega * (t - t0)) / (2 * sp.pi),
        (k, -sp.oo, sp.oo),
    )
    return UnifiedTransformResult(
        solution=solution,
        dispersion_relation=omega,
        domain="half_line",
        method_family="unified_transform_half_line_formal",
        is_formal=True,
        notes=f"Formal half-line spectral representation for {profile.family_name} with an unresolved boundary transform G(k,t).",
        profile=profile,
    )


def solve_unified_transform_half_line(
    eq: sp.Equality | sp.Expr,
    initial_condition: sp.Equality,
    boundary_conditions: Sequence[sp.Equality],
    u: sp.Function,
    vars: tuple[sp.Symbol, sp.Symbol],
) -> UnifiedTransformResult:
    """Solve a minimal class of half-line scalar evolution PDEs."""
    profile = recognize_evolution_pde(eq, u, vars)
    if profile is None:
        raise NotImplementedError(
            "Could not recognize a scalar constant-coefficient evolution PDE"
        )

    explicit = _solve_half_line_advection(
        profile, initial_condition, boundary_conditions, u, vars
    )
    if explicit is not None:
        return explicit
    explicit = _solve_half_line_heat_dirichlet(
        profile, initial_condition, boundary_conditions, u, vars
    )
    if explicit is not None:
        return explicit
    explicit = _solve_half_line_schrodinger_dirichlet_zero(
        profile, initial_condition, boundary_conditions, u, vars
    )
    if explicit is not None:
        return explicit
    return _formal_half_line_representation(
        profile, initial_condition, boundary_conditions, u, vars
    )


def solve_unified_transform(
    eq: sp.Equality | sp.Expr,
    u: sp.Function,
    vars: tuple[sp.Symbol, sp.Symbol],
    *,
    initial_condition: Optional[sp.Equality] = None,
    boundary_conditions: Optional[Sequence[sp.Equality]] = None,
    domain: str = "whole_line",
) -> UnifiedTransformResult:
    """Top-level unified-transform entry point."""
    if initial_condition is None:
        raise ValueError(
            "initial_condition is required for the unified-transform solvers"
        )
    boundary_conditions = list(boundary_conditions or [])
    if domain == "whole_line":
        return solve_unified_transform_whole_line(eq, initial_condition, u, vars)
    if domain == "half_line":
        return solve_unified_transform_half_line(
            eq, initial_condition, boundary_conditions, u, vars
        )
    raise NotImplementedError(f"Unsupported unified-transform domain: {domain}")
