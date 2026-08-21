from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp

from .classical_symbolic_helpers import _as_zero_expr, _dep_and_vars


def _core():
    from . import classical_methods as core_mod

    return core_mod


@dataclass(frozen=True)
class PDEInitialCondition1D:
    location: sp.Expr
    value: sp.Expr | Callable
    derivative_order: int = 0


@dataclass(frozen=True)
class PDEBoundaryCondition1D:
    location: sp.Expr
    kind: str  # dirichlet | neumann | robin
    value: sp.Expr | Callable
    coefficient: sp.Expr = 1


@dataclass(frozen=True)
class PDEVerificationReport:
    pde_residual: sp.Expr
    initial_residuals: tuple[sp.Expr, ...]
    boundary_residuals: tuple[sp.Expr, ...]
    assumptions: object
    verified: bool


@dataclass(frozen=True)
class PDEFamilyRecognition:
    family: str
    normalized_equation: sp.Equality
    parameters: dict


@dataclass(frozen=True)
class SeparationResult:
    family: str
    ansatz: sp.Equality
    separated_odes: tuple[sp.Equality, ...]
    separation_constants: tuple[sp.Symbol, ...]
    basis_hint: str | None
    details: dict


def _normalize_condition_dicts(ics=None, bcs=None):
    norm_ics = []
    norm_bcs = []

    def _consume_ic_eq(eq):
        if not isinstance(eq, sp.Equality):
            return
        lhs = eq.lhs
        if getattr(lhs, "is_Function", False) and len(getattr(lhs, "args", ())) >= 2:
            loc = lhs.args[-1]
            if not loc.has(*lhs.args):
                norm_ics.append(PDEInitialCondition1D(loc, eq.rhs, 0))
        elif (
            isinstance(lhs, sp.Derivative)
            and getattr(lhs.expr, "is_Function", False)
            and len(getattr(lhs.expr, "args", ())) >= 2
        ):
            args = lhs.expr.args
            t = args[-1]
            order_t = 0
            for var, count in lhs.variable_count:
                if var == t:
                    order_t = count
            if order_t:
                norm_ics.append(
                    PDEInitialCondition1D(t if not t.has(*args) else 0, eq.rhs, order_t)
                )

    def _consume_bc_eq(eq):
        if not isinstance(eq, sp.Equality):
            return
        lhs = eq.lhs
        if getattr(lhs, "is_Function", False) and len(getattr(lhs, "args", ())) >= 2:
            x = lhs.args[0]
            if not x.has(*lhs.args):
                norm_bcs.append(PDEBoundaryCondition1D(x, "dirichlet", eq.rhs))
        elif isinstance(lhs, sp.Derivative) and getattr(lhs.expr, "is_Function", False):
            x = lhs.expr.args[0]
            dx_order = 0
            for var, count in lhs.variable_count:
                if var == x:
                    dx_order = count
            if dx_order == 1:
                norm_bcs.append(
                    PDEBoundaryCondition1D(x if not x.has(*lhs.expr.args) else 0, "neumann", eq.rhs)
                )

    if isinstance(ics, dict):
        if "initial_profile" in ics:
            norm_ics.append(
                PDEInitialCondition1D(ics.get("curve_value", 0), ics["initial_profile"], 0)
            )
        if "initial_displacement" in ics:
            norm_ics.append(
                PDEInitialCondition1D(ics.get("curve_value", 0), ics["initial_displacement"], 0)
            )
        if "initial_velocity" in ics:
            norm_ics.append(
                PDEInitialCondition1D(ics.get("curve_value", 0), ics["initial_velocity"], 1)
            )
        for key in ("equations", "equation", "initial_equation"):
            payload = ics.get(key)
            if isinstance(payload, (list, tuple)):
                for eq in payload:
                    _consume_ic_eq(eq)
            elif payload is not None:
                _consume_ic_eq(payload)
    elif isinstance(ics, (list, tuple)):
        for ic in ics:
            if isinstance(ic, PDEInitialCondition1D):
                norm_ics.append(ic)
            else:
                _consume_ic_eq(ic)

    if isinstance(bcs, dict):
        btype = bcs.get("type")
        if btype == "dirichlet_homogeneous_interval":
            L = bcs.get("length", sp.pi)
            norm_bcs.extend(
                [
                    PDEBoundaryCondition1D(0, "dirichlet", 0),
                    PDEBoundaryCondition1D(L, "dirichlet", 0),
                ]
            )
        elif btype == "neumann_homogeneous_interval":
            L = bcs.get("length", sp.pi)
            norm_bcs.extend(
                [
                    PDEBoundaryCondition1D(0, "neumann", 0),
                    PDEBoundaryCondition1D(L, "neumann", 0),
                ]
            )
        elif btype == "robin_interval":
            L = bcs.get("length", sp.pi)
            left = bcs.get("left", (1, 0))
            right = bcs.get("right", (1, 0))
            lcoef = left[0] if isinstance(left, (tuple, list)) else 1
            rcoef = right[0] if isinstance(right, (tuple, list)) else 1
            lval = left[1] if isinstance(left, (tuple, list)) and len(left) > 1 else 0
            rval = right[1] if isinstance(right, (tuple, list)) and len(right) > 1 else 0
            norm_bcs.extend(
                [
                    PDEBoundaryCondition1D(0, "robin", lval, coefficient=lcoef),
                    PDEBoundaryCondition1D(L, "robin", rval, coefficient=rcoef),
                ]
            )
        for key in ("equations", "equation"):
            payload = bcs.get(key)
            if isinstance(payload, (list, tuple)):
                for eq in payload:
                    _consume_bc_eq(eq)
            elif payload is not None:
                _consume_bc_eq(payload)
    elif isinstance(bcs, (list, tuple)):
        for bc in bcs:
            if isinstance(bc, PDEBoundaryCondition1D):
                norm_bcs.append(bc)
            else:
                _consume_bc_eq(bc)
    return tuple(norm_ics), tuple(norm_bcs)


def _canonical_linear_pde_1d_xt(eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if len(vars_) != 2:
        raise ValueError("Expected exactly two variables for canonical 1+1 PDE detection.")
    x, t = vars_
    zero = _as_zero_expr(eq_or_expr)
    ux = sp.diff(uexpr, x)
    ut = sp.diff(uexpr, t)
    uxx = sp.diff(uexpr, x, 2)
    utt = sp.diff(uexpr, t, 2)

    c_uxx = sp.expand(sp.diff(zero, uxx))
    c_utt = sp.expand(sp.diff(zero, utt))
    c_ut = sp.expand(sp.diff(zero, ut))
    c_ux = sp.expand(sp.diff(zero, ux))
    residual = sp.expand(zero.subs({uxx: 0, utt: 0, ut: 0, ux: 0, uexpr: 0}))
    c_u = sp.expand(sp.diff(zero.subs({uxx: 0, utt: 0, ut: 0, ux: 0}), uexpr))
    test = sp.expand(
        zero - c_uxx * uxx - c_utt * utt - c_ut * ut - c_ux * ux - c_u * uexpr - residual
    )
    if test != 0:
        return None

    if (
        sp.simplify(c_ut) != 0
        and sp.simplify(c_uxx) != 0
        and sp.simplify(c_utt) == 0
        and sp.simplify(c_ux) == 0
    ):
        kappa = sp.simplify(-c_uxx / c_ut)
        q = sp.simplify(-c_u / c_ut)
        rest = sp.simplify(-residual / c_ut)
        return {
            "kind": "heat_like",
            "vars": (x, t),
            "u": uexpr,
            "kappa": kappa,
            "q": q,
            "forcing": rest,
            "normalized": sp.Eq(ut, kappa * uxx + q * uexpr + rest),
        }

    if (
        sp.simplify(c_utt) != 0
        and sp.simplify(c_uxx) != 0
        and sp.simplify(c_ut) == 0
        and sp.simplify(c_ux) == 0
    ):
        c2 = sp.simplify(-c_uxx / c_utt)
        q = sp.simplify(-c_u / c_utt)
        rest = sp.simplify(-residual / c_utt)
        return {
            "kind": "wave_like",
            "vars": (x, t),
            "u": uexpr,
            "c2": c2,
            "q": q,
            "forcing": rest,
            "normalized": sp.Eq(utt, c2 * uxx + q * uexpr + rest),
        }

    if sp.simplify(c_ut) == 0 and sp.simplify(c_utt) == 0 and sp.simplify(c_uxx) != 0:
        y = t
        uyy = sp.diff(uexpr, y, 2)
        c_uyy = sp.expand(sp.diff(zero, uyy))
        c_uy = sp.expand(sp.diff(zero, sp.diff(uexpr, y)))
        residual2 = sp.expand(
            zero.subs({uxx: 0, uyy: 0, sp.diff(uexpr, x): 0, sp.diff(uexpr, y): 0, uexpr: 0})
        )
        c_u2 = sp.expand(
            sp.diff(zero.subs({uxx: 0, uyy: 0, sp.diff(uexpr, x): 0, sp.diff(uexpr, y): 0}), uexpr)
        )
        test2 = sp.expand(
            zero
            - c_uxx * uxx
            - c_uyy * uyy
            - c_ux * sp.diff(uexpr, x)
            - c_uy * sp.diff(uexpr, y)
            - c_u2 * uexpr
            - residual2
        )
        if (
            test2 == 0
            and sp.simplify(c_ux) == 0
            and sp.simplify(c_uy) == 0
            and sp.simplify(c_uyy) != 0
        ):
            ay = sp.simplify(c_uyy / c_uxx)
            lam = sp.simplify(c_u2 / c_uxx)
            rest2 = sp.simplify(residual2 / c_uxx)
            return {
                "kind": "laplace_helmholtz_like",
                "vars": (x, y),
                "u": uexpr,
                "ay": ay,
                "lambda0": lam,
                "forcing": rest2,
                "normalized": sp.Eq(uxx + ay * uyy + lam * uexpr + rest2, 0),
            }
    return None


def _recognize_base_family(eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True):
    """Recognize a few important scalar PDE families in 1+1 or 2D."""
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    zero_eq = _core().canonicalize_pde_problem(eq_or_expr, uexpr, vars_).equation
    zero = _as_zero_expr(zero_eq)
    if len(vars_) == 2:
        x, t = vars_
        ux = sp.diff(uexpr, x)
        ut = sp.diff(uexpr, t)
        uxx = sp.diff(uexpr, x, 2)
        utt = sp.diff(uexpr, t, 2)
        # Burgers / viscous Burgers: u_t + u u_x = nu u_xx
        if sp.simplify(sp.diff(zero, ut) - 1) == 0 and sp.simplify(sp.diff(zero, uxx)) != 0:
            coeff_uxx = -sp.expand(sp.diff(zero, uxx))
            rest = sp.expand(zero - ut + coeff_uxx * uxx)
            if sp.simplify(rest - uexpr * ux) == 0:
                fam = "viscous_burgers" if sp.simplify(coeff_uxx) != 0 else "inviscid_burgers"
                return PDEFamilyRecognition(fam, zero_eq, {"viscosity": coeff_uxx})
        # Telegraph: u_tt + a u_t = c^2 u_xx + b u
        if sp.simplify(sp.diff(zero, utt) - 1) == 0 and sp.simplify(sp.diff(zero, uxx)) != 0:
            c2 = -sp.expand(sp.diff(zero, uxx))
            a = sp.expand(sp.diff(zero.subs({utt: 0, uxx: 0}), ut))
            b = -sp.expand(sp.diff(zero.subs({utt: 0, uxx: 0, ut: 0}), uexpr))
            rest = sp.expand(zero - utt + c2 * uxx - a * ut + b * uexpr)
            if sp.simplify(rest) == 0:
                return PDEFamilyRecognition(
                    "telegraph_like", zero_eq, {"wave_speed_sq": c2, "damping": a, "mass": b}
                )
        # Klein-Gordon/Helmholtz/Laplace/Poisson style
        cls = None
        try:
            cls = _core().classify_linear_second_order_pde(
                zero_eq, uexpr, vars_, assumptions=assumptions
            )
        except Exception:
            cls = None
        if cls is not None and cls.classification in {
            "elliptic",
            "hyperbolic",
            "parabolic",
            "ultrahyperbolic",
        }:
            params = {"classification": cls.classification, "matrix": cls.coefficient_matrix}
            if cls.classification == "elliptic" and len(vars_) == 2:
                params["family_hint"] = "laplace_helmholtz_poisson_like"
            return PDEFamilyRecognition("linear_second_order", zero_eq, params)
    return None


def recognize_pde_family(eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True):
    base = _recognize_base_family(eq_or_expr, dep_expr_or_func, indep_vars, assumptions=assumptions)
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    zero_eq = _core().canonicalize_pde_problem(eq_or_expr, uexpr, vars_).equation
    zero = _as_zero_expr(zero_eq)
    if len(vars_) == 2:
        x, t = vars_
        ux = sp.diff(uexpr, x)
        ut = sp.diff(uexpr, t)
        uxx = sp.diff(uexpr, x, 2)
        utt = sp.diff(uexpr, t, 2)
        # Prefer Black-Scholes-like before generic advection-diffusion-reaction.
        try:
            c_ut = sp.expand(sp.diff(zero, ut))
            c_uxx = sp.expand(sp.diff(zero, uxx))
            c_ux = sp.expand(sp.diff(zero, ux))
            c_u = sp.expand(sp.diff(zero.subs({ut: 0, uxx: 0, ux: 0}), uexpr))
            rest = sp.expand(zero - c_ut * ut - c_uxx * uxx - c_ux * ux - c_u * uexpr)
            if (
                rest == 0
                and c_ut != 0
                and any(
                    pow_.is_Integer and int(pow_) >= 2
                    for pow_ in [sp.degree(c_uxx, x)]
                    if pow_ is not None
                )
            ):
                if x in c_uxx.free_symbols or x in c_ux.free_symbols:
                    return PDEFamilyRecognition(
                        "black_scholes_like",
                        zero_eq,
                        {
                            "time_coeff": c_ut,
                            "diffusion_coeff": c_uxx,
                            "drift_coeff": c_ux,
                            "reaction_coeff": c_u,
                        },
                    )
        except Exception:
            pass
        if base is not None and base.family != "linear_second_order":
            return base
        # Linear transport-reaction-diffusion
        try:
            A = sp.expand(sp.diff(zero, ut))
            B = sp.expand(sp.diff(zero, ux))
            C = sp.expand(sp.diff(zero, uxx))
            D = sp.expand(sp.diff(zero.subs({ut: 0, ux: 0, uxx: 0}), uexpr))
            rest = sp.expand(zero - A * ut - B * ux - C * uxx - D * uexpr)
            if (
                A != 0
                and rest == 0
                and all(v.free_symbols.isdisjoint({uexpr}) for v in (A, B, C, D))
            ):
                if C != 0:
                    return PDEFamilyRecognition(
                        "advection_diffusion_reaction",
                        zero_eq,
                        {
                            "time_coeff": A,
                            "advection": B,
                            "diffusion": -C / A if A != 0 else C,
                            "reaction": D,
                        },
                    )
                return PDEFamilyRecognition(
                    "transport_reaction", zero_eq, {"time_coeff": A, "transport": B, "reaction": D}
                )
        except Exception:
            pass
        try:
            A = sp.expand(sp.diff(zero, ut))
            C = sp.expand(sp.diff(zero, uxx))
            if A != 0 and C != 0:
                rem = sp.expand(zero - A * ut - C * uxx)
                if rem.free_symbols.isdisjoint({ux, ut, uxx, utt}):
                    return PDEFamilyRecognition(
                        "reaction_diffusion_like",
                        zero_eq,
                        {
                            "time_coeff": A,
                            "diffusion_coeff": -C / A if A != 0 else C,
                            "reaction_term": sp.expand(-rem / A),
                        },
                    )
        except Exception:
            pass
    return base


def detect_scalar_conservation_law_family(eq_or_expr, dep_expr_or_func, indep_vars=None):
    cons = _core().detect_conservation_law_1d(eq_or_expr, dep_expr_or_func, indep_vars)
    uexpr = cons.dep_function
    x, t = cons.indep_vars
    flux = cons.flux
    u = sp.Symbol("u")
    try:
        flux_u = sp.simplify(sp.diff(flux.subs(uexpr, u), u))
    except Exception:
        flux_u = None
    family = "scalar_conservation_law"
    if flux_u is not None and not flux_u.has(x, t):
        if sp.simplify(flux.subs(uexpr, u) - u**2 / 2) == 0:
            family = "inviscid_burgers"
    return PDEFamilyRecognition(
        family, cons.normalized_equation, {"flux": flux, "density": cons.density}
    )


def detect_burgers_family(eq_or_expr, dep_expr_or_func, indep_vars=None):
    """
    Detect inviscid or viscous Burgers-family equations in 2 variables:

        u_t + u u_x = 0
        u_t + u u_x = nu u_xx
        u_t + alpha u u_x = nu u_xx + forcing(x,t)
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if len(vars_) != 2:
        raise ValueError("Expected two variables (x,t).")
    x, t = vars_
    zero = _as_zero_expr(eq_or_expr)
    ux = sp.diff(uexpr, x)
    ut = sp.diff(uexpr, t)
    uxx = sp.diff(uexpr, x, 2)

    A = sp.expand(sp.diff(zero, ut))
    B = sp.expand(sp.diff(zero, uxx))
    Cux = sp.expand(sp.diff(zero, ux))
    rest = sp.expand(zero - A * ut - B * uxx - Cux * ux)

    if sp.simplify(A) == 0:
        return None
    alpha = sp.simplify(Cux / (A * uexpr)) if sp.simplify(uexpr) != 0 else None
    if alpha is None:
        return None

    # Require Cux proportional to u.
    usym = sp.Symbol("_uburg")
    Cux_hold = sp.expand(Cux.subs({uexpr: usym}))
    if sp.simplify(sp.diff(Cux_hold, usym, 2)) != 0:
        return None
    alpha_hold = sp.simplify(Cux_hold / usym) if usym in Cux_hold.free_symbols else sp.Integer(0)
    if alpha_hold.has(x, t):
        return None

    nu = sp.simplify(-B / A)
    forcing = sp.simplify(-rest / A)
    family = "viscous_burgers" if sp.simplify(nu) != 0 else "inviscid_burgers"
    return PDEFamilyRecognition(
        family, sp.Eq(sp.expand(zero), 0), {"alpha": alpha_hold, "nu": nu, "forcing": forcing}
    )
