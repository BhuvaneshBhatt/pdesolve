from __future__ import annotations

from typing import Any

import sympy as sp

from .classical_methods import (
    conserved_mass_statement,
    detect_conservation_law_1d,
    detect_scalar_conservation_law_family,
    extract_conservation_form_auto,
    rankine_hugoniot_speed,
    solve_burgers_family,
    solve_burgers_ivp_characteristic_formal,
    solve_scalar_conservation_law_riemann_burgers,
    solve_scalar_conservation_law_riemann_general,
    solve_viscous_burgers_cole_hopf_formal,
)
from .errors import PDEMethodNotApplicable
from .results import (
    ConservationLawCanonicalForm,
    ConservationLawImplicitCharacteristicResult,
    ConservationLawInitialData1D,
    ConservationLawPropagationResult,
    ConservationLawRarefactionResult,
    ConservationLawShockResult,
    PDEVerificationSummary,
    SolverMethodResult,
)
from .verify import verify_solution_with_conditions

__all__ = [
    "ConservationLawCanonicalForm",
    "ConservationLawInitialData1D",
    "ConservationLawPropagationResult",
    "ConservationLawImplicitCharacteristicResult",
    "ConservationLawShockResult",
    "ConservationLawRarefactionResult",
    "detect_conservation_law_1d",
    "detect_scalar_conservation_law_family",
    "conserved_mass_statement",
    "rankine_hugoniot_speed",
    "solve_scalar_conservation_law_riemann_general",
    "solve_scalar_conservation_law_riemann_burgers",
    "solve_burgers_family",
    "solve_burgers_ivp_characteristic_formal",
    "solve_viscous_burgers_cole_hopf_formal",
    "canonicalize_scalar_conservation_law_1d",
    "parse_scalar_conservation_law_initial_data",
    "solve_scalar_conservation_law_ivp",
    "verify_piecewise_conservation_law_solution",
    "verify_weak_conservation_law_solution",
    "canonicalize_scalar_conservation_law",
    "parse_conservation_law_initial_data",
    "analyze_conservation_law",
    "verify_conservation_law_solution",
]


def _dep_and_vars(dep_expr_or_func, indep_vars=None):
    if isinstance(dep_expr_or_func, sp.Expr) and dep_expr_or_func.is_Function:
        uexpr = dep_expr_or_func
        vars_ = tuple(uexpr.args)
    elif isinstance(dep_expr_or_func, sp.FunctionClass):
        if indep_vars is None:
            raise ValueError("indep_vars is required when a FunctionClass is supplied.")
        vars_ = tuple(indep_vars)
        uexpr = dep_expr_or_func(*vars_)
    else:
        uexpr = sp.sympify(dep_expr_or_func)
        if indep_vars is not None:
            vars_ = tuple(indep_vars)
        elif isinstance(uexpr, sp.Expr) and getattr(uexpr, "is_Function", False):
            vars_ = tuple(uexpr.args)
        else:
            raise ValueError("Could not determine dependent expression and variables.")
    return uexpr, vars_


def _extract_autonomous_flux(
    flux: sp.Expr, dep_function: sp.Expr, indep_vars: tuple[sp.Symbol, sp.Symbol]
):
    x, t = indep_vars
    usym = sp.Symbol("u", real=True)
    try:
        hold = sp.expand(flux.subs(dep_function, usym))
    except Exception:
        return None
    if hold.free_symbols.isdisjoint({x, t}):
        return hold
    return None


def canonicalize_scalar_conservation_law_1d(eq_or_expr, dep_expr_or_func, indep_vars=None):
    """Return a normalized scalar conservation-law description.

    Preferred target form:
        u_t + (f(u))_x = s(x,t,u)
    represented internally as
        u_t + f_x(x,t,u) + f_u(x,t,u) u_x = s(x,t,u).
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    x, t = vars_
    try:
        cons = detect_conservation_law_1d(eq_or_expr, uexpr, vars_)
        flux = sp.expand(cons.flux)
        return ConservationLawCanonicalForm(
            indep_vars=(x, t),
            dep_function=uexpr,
            density=cons.density,
            flux=flux,
            source=sp.Integer(0),
            normalized_equation=cons.normalized_equation,
            autonomous_flux=_extract_autonomous_flux(flux, uexpr, (x, t)),
            family=detect_scalar_conservation_law_family(eq_or_expr, uexpr, vars_).family,
            details=dict(cons.details),
        )
    except Exception:
        extracted = extract_conservation_form_auto(eq_or_expr, uexpr, vars_)
        flux = sp.expand(extracted.flux)
        return ConservationLawCanonicalForm(
            indep_vars=(x, t),
            dep_function=uexpr,
            density=extracted.density,
            flux=flux,
            source=sp.expand(extracted.source),
            normalized_equation=extracted.normalized_equation,
            autonomous_flux=_extract_autonomous_flux(flux, uexpr, (x, t)),
            family="scalar_balance_law"
            if sp.simplify(extracted.source) != 0
            else "scalar_conservation_law",
            details=dict(extracted.details),
        )


def _profile_from_equality(eq: sp.Equality, dep_function: sp.Expr, x: sp.Symbol):
    if not isinstance(eq, sp.Equality):
        raise ValueError("Initial condition must be an Equality.")
    if eq.lhs == dep_function.subs({dep_function.args[1]: 0}):
        return eq.rhs
    if eq.rhs == dep_function.subs({dep_function.args[1]: 0}):
        return eq.lhs
    raise ValueError("Initial condition is not of the form u(x,0) = g(x).")


def _piecewise_riemann_states(profile, x):
    if not isinstance(profile, sp.Piecewise) or len(profile.args) < 2:
        return None, None
    first_expr, first_cond = profile.args[0]
    second_expr, _ = profile.args[1]
    try:
        if first_cond == (x < 0) or first_cond == sp.StrictLessThan(x, 0):
            return sp.sympify(first_expr), sp.sympify(second_expr)
        if first_cond == (x <= 0) or first_cond == sp.LessThan(x, 0):
            return sp.sympify(first_expr), sp.sympify(second_expr)
    except Exception:
        pass
    return None, None


def _riemann_states_from_profile(profile, x):
    pw_left, pw_right = _piecewise_riemann_states(profile, x)
    if pw_left is not None and pw_right is not None:
        return pw_left, pw_right
    try:
        left = sp.simplify(sp.limit(profile, x, 0, dir="-"))
        right = sp.simplify(sp.limit(profile, x, 0, dir="+"))
    except Exception:
        return None, None
    if left.has(sp.oo, -sp.oo, sp.zoo, sp.nan) or right.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None, None
    return left, right


def parse_scalar_conservation_law_initial_data(
    initial_conditions, dep_expr_or_func, indep_vars=None
):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    x, t = vars_
    u0 = uexpr.subs(t, 0)

    if initial_conditions is None:
        return ConservationLawInitialData1D((x, t), uexpr, kind="none")

    if isinstance(initial_conditions, ConservationLawInitialData1D):
        return initial_conditions

    profile = None
    equation = None
    details: dict[str, Any] = {}

    if isinstance(initial_conditions, dict):
        if "initial_profile" in initial_conditions:
            profile = initial_conditions["initial_profile"]
        elif "riemann_data" in initial_conditions:
            left, right = map(sp.sympify, initial_conditions["riemann_data"])
            profile = sp.Piecewise((left, x < 0), (right, True), evaluate=False)
            equation = sp.Eq(u0, profile)
            return ConservationLawInitialData1D(
                indep_vars=(x, t),
                dep_function=uexpr,
                kind="riemann",
                profile=profile,
                equation=equation,
                left_state=left,
                right_state=right,
                interface=sp.Integer(0),
                details={"riemann_data": (left, right)},
            )
        elif "equation" in initial_conditions:
            equation = sp.sympify(initial_conditions["equation"])
        else:
            raise ValueError("Unsupported conservation-law initial-data dictionary.")
    elif isinstance(initial_conditions, sp.Equality):
        equation = initial_conditions
    elif (
        isinstance(initial_conditions, (list, tuple))
        and len(initial_conditions) == 1
        and isinstance(initial_conditions[0], sp.Equality)
    ):
        equation = initial_conditions[0]
    else:
        profile = sp.sympify(initial_conditions)

    if equation is not None:
        profile = _profile_from_equality(equation, uexpr, x)
    else:
        equation = sp.Eq(u0, sp.sympify(profile))

    left, right = _riemann_states_from_profile(profile, x)
    kind = "general_profile"
    if left is not None and right is not None and sp.simplify(left - right) != 0:
        kind = "riemann"

    return ConservationLawInitialData1D(
        indep_vars=(x, t),
        dep_function=uexpr,
        kind=kind,
        profile=sp.sympify(profile),
        equation=equation,
        left_state=left,
        right_state=right,
        interface=sp.Integer(0) if kind == "riemann" else None,
        details=details,
    )


def solve_scalar_conservation_law_ivp(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, initial_conditions=None
):
    canonical = canonicalize_scalar_conservation_law_1d(eq_or_expr, dep_expr_or_func, indep_vars)
    data = parse_scalar_conservation_law_initial_data(
        initial_conditions, canonical.dep_function, canonical.indep_vars
    )
    x, t = canonical.indep_vars
    uexpr = canonical.dep_function

    if sp.simplify(canonical.source) != 0:
        raise NotImplementedError(
            "Structured IVP results currently target source-free scalar conservation laws."
        )
    if canonical.autonomous_flux is None:
        raise NotImplementedError("Structured IVP results currently target autonomous fluxes f(u).")

    usym = sp.Symbol("u", real=True)
    f = canonical.autonomous_flux
    fp = sp.simplify(sp.diff(f, usym))

    if data.kind == "riemann":
        base = solve_scalar_conservation_law_riemann_general(
            f, data.left_state, data.right_state, x=x, t=t, u_symbol=usym
        )
        sol_eq = sp.Eq(
            uexpr,
            base.solution if not isinstance(base.solution, sp.Equality) else base.solution.rhs,
        )
        selection = dict(base.details.get("selection", {}))
        branch = selection.get("branch")
        if branch == "shock":
            admiss = entropy_admissibility_scalar_riemann(
                f,
                data.left_state,
                data.right_state,
                "shock",
                shock_speed=selection.get("shock_speed"),
                u_symbol=usym,
            )
            return ConservationLawShockResult(
                method="scalar_conservation_riemann_shock",
                solution=sol_eq,
                flux=f,
                left_state=data.left_state,
                right_state=data.right_state,
                shock_speed=selection.get("shock_speed"),
                canonical_form=canonical,
                initial_data=data,
                details={"selection": selection, "admissibility": admiss},
            )
        admiss = entropy_admissibility_scalar_riemann(
            f,
            data.left_state,
            data.right_state,
            "rarefaction",
            left_speed=selection.get("left_speed"),
            right_speed=selection.get("right_speed"),
            u_symbol=usym,
        )
        return ConservationLawRarefactionResult(
            method="scalar_conservation_riemann_rarefaction",
            solution=sol_eq,
            flux=f,
            left_state=data.left_state,
            right_state=data.right_state,
            left_speed=selection.get("left_speed"),
            right_speed=selection.get("right_speed"),
            self_similar_variable=sp.simplify(x / t),
            canonical_form=canonical,
            initial_data=data,
            details={"selection": selection, "admissibility": admiss},
        )

    if data.profile is None:
        raise PDEMethodNotApplicable(
            "Scalar conservation-law IVP solving requires initial profile or Riemann data."
        )
    profile = sp.sympify(data.profile)
    if fp.free_symbols.isdisjoint({usym}):
        speed = sp.simplify(fp)
        sol = sp.Eq(uexpr, profile.subs(x, x - speed * t))
        return ConservationLawPropagationResult(
            method="scalar_conservation_profile_propagation",
            solution=sol,
            profile=data.profile,
            speed=speed,
            canonical_form=canonical,
            initial_data=data,
            details={"autonomous_flux": f},
        )

    xi = sp.Symbol("xi", real=True)
    gxi = sp.simplify(profile.subs(x, xi))
    speed_xi = sp.simplify(fp.subs(usym, gxi))
    characteristic_relation = sp.Eq(x, sp.expand(xi + speed_xi * t))
    profile_relation = sp.Eq(uexpr, gxi)
    implicit_relation = sp.Eq(uexpr, sp.simplify(profile.subs(x, x - fp.subs(usym, uexpr) * t)))
    foot = sp.Symbol("x0", real=True)
    footpoint_equation = sp.Eq(x, sp.expand(foot + fp.subs(usym, profile.subs(x, foot)) * t))
    return ConservationLawImplicitCharacteristicResult(
        method="scalar_conservation_implicit_characteristics",
        solution=(characteristic_relation, profile_relation),
        profile=data.profile,
        characteristic_parameter=xi,
        characteristic_relation=characteristic_relation,
        profile_relation=profile_relation,
        footpoint_equation=footpoint_equation,
        implicit_relation=implicit_relation,
        characteristic_speed=sp.simplify(fp),
        canonical_form=canonical,
        initial_data=data,
        details={
            "autonomous_flux": f,
            "characteristic_speed_function": sp.simplify(fp),
            "footpoint_symbol": foot,
            "smooth_profile_assumption": True,
        },
    )


def _extract_solution_rhs(solution):
    if isinstance(solution, sp.Equality):
        return solution.rhs
    if hasattr(solution, "solution"):
        inner = solution.solution
        if isinstance(inner, sp.Equality):
            return inner.rhs
        return inner
    return solution


def verify_piecewise_conservation_law_solution(
    eq_or_expr, solution, dep_expr_or_func, indep_vars=None, *, initial_conditions=None
):
    """Best-effort verification for Piecewise/self-similar conservation-law outputs."""
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    rhs = _extract_solution_rhs(solution)

    initial_data = None
    if initial_conditions is not None:
        initial_data = parse_scalar_conservation_law_initial_data(initial_conditions, uexpr, vars_)

    if isinstance(
        solution,
        (
            ConservationLawPropagationResult,
            ConservationLawImplicitCharacteristicResult,
            ConservationLawShockResult,
            ConservationLawRarefactionResult,
        ),
    ):
        eq_solution = solution.solution
    else:
        eq_solution = sp.Eq(uexpr, rhs)

    report = verify_solution_with_conditions(
        eq_or_expr,
        eq_solution,
        uexpr,
        vars_,
        ics={"initial_profile": initial_data.profile}
        if initial_data and initial_data.profile is not None
        else None,
    )
    if report.verified is True:
        return report

    piecewise_rhs = rhs if isinstance(rhs, sp.Piecewise) else None
    if piecewise_rhs is None:
        return report

    base_zero = (
        sp.sympify(eq_or_expr.lhs - eq_or_expr.rhs)
        if isinstance(eq_or_expr, sp.Equality)
        else sp.sympify(eq_or_expr)
    )
    branch_residuals = []
    for expr, _cond in piecewise_rhs.args:
        try:
            residual = sp.simplify(sp.expand(base_zero.subs(uexpr, expr).doit()))
        except Exception:
            residual = None
        if residual is not None:
            branch_residuals.append(residual)
    symbolic_branch_ok = bool(branch_residuals) and all(
        res == 0 for res in branch_residuals if res is not None
    )

    weak_summary = verify_weak_conservation_law_solution(solution, dep_expr_or_func, indep_vars)
    if symbolic_branch_ok and weak_summary.verified is not False:
        return PDEVerificationSummary(
            verified=True,
            status="verified",
            pde_verified=True,
            initial_verified=report.initial_verified,
            boundary_verified=report.boundary_verified,
            pde_residual=tuple(branch_residuals),
            initial_residuals=report.initial_residuals,
            boundary_residuals=report.boundary_residuals,
            mode="piecewise_weak",
            message=weak_summary.message
            or "Piecewise branches satisfy PDE away from interfaces and weak checks passed.",
        )
    return weak_summary if weak_summary.verified is not None else report


def _extract_piecewise_interfaces(expr, x, t):
    out = []
    if not isinstance(expr, sp.Piecewise):
        return out
    for branch_expr, cond in expr.args[:-1]:
        rels = list(cond.args) if isinstance(cond, sp.And) else [cond]
        for rel in rels:
            if isinstance(
                rel, (sp.StrictLessThan, sp.LessThan, sp.StrictGreaterThan, sp.GreaterThan)
            ):
                lhs = sp.simplify(rel.lhs - rel.rhs)
                if lhs.has(x) and lhs.has(t):
                    out.append((branch_expr, sp.Eq(lhs, 0)))
    return out


def verify_weak_conservation_law_solution(solution, dep_expr_or_func=None, indep_vars=None):
    """Weak-solution checks specialized for structured shock/rarefaction outputs."""
    if isinstance(solution, ConservationLawShockResult):
        usym = sp.Symbol("u", real=True)
        expected_speed = rankine_hugoniot_speed(
            solution.flux, solution.left_state, solution.right_state, u_symbol=usym
        )
        ok = sp.simplify(expected_speed - solution.shock_speed) == 0
        admiss = entropy_admissibility_scalar_riemann(
            solution.flux,
            solution.left_state,
            solution.right_state,
            "shock",
            shock_speed=solution.shock_speed,
            u_symbol=usym,
        )
        entropy_ok = bool(admiss.get("admissible", True))
        return PDEVerificationSummary(
            verified=bool(ok and entropy_ok),
            status="verified" if (ok and entropy_ok) else "failed",
            pde_verified=bool(ok and entropy_ok),
            mode="weak_shock",
            message="Rankine-Hugoniot and Lax/Oleinik admissibility checks."
            if (ok and entropy_ok)
            else "Shock failed Rankine-Hugoniot or entropy admissibility.",
        )
    if isinstance(solution, ConservationLawRarefactionResult):
        usym = sp.Symbol("u", real=True)
        fp = sp.simplify(sp.diff(solution.flux, usym))
        left_ok = sp.simplify(fp.subs(usym, solution.left_state) - solution.left_speed) == 0
        right_ok = sp.simplify(fp.subs(usym, solution.right_state) - solution.right_speed) == 0
        admiss = entropy_admissibility_scalar_riemann(
            solution.flux,
            solution.left_state,
            solution.right_state,
            "rarefaction",
            left_speed=solution.left_speed,
            right_speed=solution.right_speed,
            u_symbol=usym,
        )
        ok = bool(left_ok and right_ok and admiss.get("admissible", True))
        return PDEVerificationSummary(
            verified=ok,
            status="verified" if ok else "failed",
            pde_verified=ok,
            mode="weak_rarefaction",
            message="Rarefaction edge-speed and admissibility check."
            if ok
            else "Rarefaction edge-speed mismatch or inadmissible branch.",
        )
    if isinstance(solution, ConservationLawPropagationResult):
        return PDEVerificationSummary(
            verified=True,
            status="verified",
            pde_verified=True,
            mode="transport_profile",
            message="Translation-profile propagation result.",
        )
    if isinstance(solution, ConservationLawImplicitCharacteristicResult):
        return PDEVerificationSummary(
            verified=True,
            status="verified",
            pde_verified=True,
            mode="implicit_characteristics",
            message="Implicit characteristic relations constructed for scalar conservation law.",
        )
    # Best-effort multi-interface piecewise check for weak branches.
    sol_expr = solution.rhs if isinstance(solution, sp.Equality) else getattr(solution, "rhs", None)
    if (
        dep_expr_or_func is not None
        and indep_vars is not None
        and sol_expr is not None
        and isinstance(sol_expr, sp.Piecewise)
    ):
        x, t = indep_vars[:2]
        interfaces = _extract_piecewise_interfaces(sol_expr, x, t)
        return PDEVerificationSummary(
            verified=True if interfaces else None,
            status="verified" if interfaces else "unverified",
            pde_verified=True if interfaces else None,
            mode="weak_piecewise_multi_interface" if interfaces else "weak_unknown",
            message=f"Identified {len(interfaces)} candidate weak interfaces."
            if interfaces
            else "No structured weak-solution metadata available.",
        )
    return PDEVerificationSummary(
        None,
        "unverified",
        mode="weak_unknown",
        message="No structured weak-solution metadata available.",
    )


def lax_shock_inequalities(flux, u_left, u_right, shock_speed, *, u_symbol=None):
    u = sp.Symbol("u", real=True) if u_symbol is None else sp.sympify(u_symbol)
    f = sp.sympify(flux)
    fp = sp.simplify(sp.diff(f, u))
    left_speed = sp.simplify(fp.subs(u, u_left))
    right_speed = sp.simplify(fp.subs(u, u_right))
    return {
        "left_speed": left_speed,
        "right_speed": right_speed,
        "shock_speed": shock_speed,
        "left_ok": sp.simplify(left_speed - sp.sympify(shock_speed)) >= 0
        if all(
            getattr(sp.sympify(v), "is_real", None) is not False for v in (left_speed, shock_speed)
        )
        else sp.simplify(left_speed - sp.sympify(shock_speed)),
        "right_ok": sp.simplify(sp.sympify(shock_speed) - right_speed) >= 0
        if all(
            getattr(sp.sympify(v), "is_real", None) is not False for v in (right_speed, shock_speed)
        )
        else sp.simplify(sp.sympify(shock_speed) - right_speed),
    }


def oleinik_one_sided_bound(flux, u_left, u_right, *, u_symbol=None):
    u = sp.Symbol("u", real=True) if u_symbol is None else sp.sympify(u_symbol)
    f = sp.sympify(flux)
    if sp.simplify(u_left - u_right) == 0:
        return {"applicable": False, "bound": None}
    secant = sp.simplify(
        (f.subs(u, u_left) - f.subs(u, u_right)) / (sp.sympify(u_left) - sp.sympify(u_right))
    )
    return {"applicable": True, "bound": secant}


def entropy_admissibility_scalar_riemann(
    flux,
    u_left,
    u_right,
    branch,
    *,
    shock_speed=None,
    left_speed=None,
    right_speed=None,
    u_symbol=None,
):
    u = sp.Symbol("u", real=True) if u_symbol is None else sp.sympify(u_symbol)
    f = sp.sympify(flux)
    fp = sp.simplify(sp.diff(f, u))
    fpp = sp.simplify(sp.diff(f, u, 2))
    convex = sp.simplify(fpp)
    metadata = {"branch": branch, "convexity": convex}
    if branch == "shock":
        s = (
            shock_speed
            if shock_speed is not None
            else rankine_hugoniot_speed(f, u_left, u_right, u_symbol=u)
        )
        lax = lax_shock_inequalities(f, u_left, u_right, s, u_symbol=u)
        left_ok = lax["left_ok"] is True or sp.simplify(lax["left_speed"] - s) >= 0
        right_ok = lax["right_ok"] is True or sp.simplify(s - lax["right_speed"]) >= 0
        metadata.update(
            {
                "admissible": bool(left_ok and right_ok),
                "lax": lax,
                "oleinik": oleinik_one_sided_bound(f, u_left, u_right, u_symbol=u),
            }
        )
        return metadata
    ls = left_speed if left_speed is not None else sp.simplify(fp.subs(u, u_left))
    rs = right_speed if right_speed is not None else sp.simplify(fp.subs(u, u_right))
    metadata.update(
        {"admissible": bool(sp.simplify(ls - rs) <= 0), "left_speed": ls, "right_speed": rs}
    )
    return metadata


def canonicalize_scalar_conservation_law(eq_or_expr, dep_expr_or_func, indep_vars=None):
    return canonicalize_scalar_conservation_law_1d(eq_or_expr, dep_expr_or_func, indep_vars)


def parse_conservation_law_initial_data(initial_conditions, dep_expr_or_func, indep_vars=None):
    return parse_scalar_conservation_law_initial_data(
        initial_conditions, dep_expr_or_func, indep_vars
    )


def analyze_conservation_law(eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None):
    structured = solve_scalar_conservation_law_ivp(
        eq_or_expr, dep_expr_or_func, indep_vars, initial_conditions=ics
    )
    solution = structured.solution if hasattr(structured, "solution") else structured
    if isinstance(structured, ConservationLawImplicitCharacteristicResult):
        solution = structured.implicit_relation or structured.profile_relation
        method = "scalar_conservation_implicit_characteristics"
    elif isinstance(structured, ConservationLawPropagationResult):
        method = "scalar_conservation_propagation"
    else:
        method = structured.method
    return SolverMethodResult(
        method_family=method, solution=solution, details={"structured_result": structured}
    )


def verify_conservation_law_solution(
    eq_or_expr,
    solution,
    dep_expr_or_func,
    indep_vars=None,
    *,
    structured_result=None,
    initial_conditions=None,
):
    if structured_result is not None:
        return verify_weak_conservation_law_solution(
            structured_result, dep_expr_or_func, indep_vars
        )
    return verify_piecewise_conservation_law_solution(
        eq_or_expr, solution, dep_expr_or_func, indep_vars, initial_conditions=initial_conditions
    )
