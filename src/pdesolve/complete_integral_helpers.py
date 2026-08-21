from __future__ import annotations

from dataclasses import dataclass
import sympy as sp
from sympy.core.function import AppliedUndef

from ._classical_shared import _dep_and_vars, _as_zero_expr, _expr_complexity


@dataclass(frozen=True)
class CompleteIntegralResult:
    method: str
    solutions: tuple[sp.Equality, ...]
    details: dict
    verification: tuple[dict, ...] = ()


@dataclass(frozen=True)
class GeneralizedClairautRecognition:
    recognized: bool
    phi: sp.Expr | None
    gradients: tuple[sp.Symbol, ...]
    details: dict


def recognize_generalized_clairaut_pde(eq_or_expr, dep_expr_or_func, indep_vars=None):
    """Recognize first-order generalized Clairaut PDEs of the form
    U = sum_i x_i * p_i + phi(p_1,...,p_n), where p_i = u_{x_i}.
    """
    _, vars_, U0, grads, F = _first_order_nonlinear_data(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    affine = sp.expand(U0 - sum(v * g for v, g in zip(vars_, grads)))
    phi = None
    ok = False
    # Either F = affine - phi or F = -affine + phi, depending on canonicalization sign.
    phi1 = sp.simplify(affine - F)
    if sp.expand(F - (affine - phi1)) == 0 and phi1.free_symbols.isdisjoint(
        set(vars_) | {U0}
    ):
        phi = sp.expand(phi1)
        ok = True
    else:
        phi2 = sp.simplify(F + affine)
        if sp.expand(F - (-affine + phi2)) == 0 and phi2.free_symbols.isdisjoint(
            set(vars_) | {U0}
        ):
            phi = sp.expand(phi2)
            ok = True
    return GeneralizedClairautRecognition(
        recognized=bool(ok),
        phi=phi if ok else None,
        gradients=tuple(grads),
        details={"F": F, "affine_part": affine},
    )


def solve_generalized_clairaut_complete_integral(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, parameter_symbols=None
):
    """Solve generalized Clairaut first-order PDEs
    u = sum_i x_i p_i + phi(p)
    by the complete integral family
    u = sum_i a_i x_i + phi(a).
    """
    uexpr, vars_, _, _, _ = _first_order_nonlinear_data(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    rec = recognize_generalized_clairaut_pde(eq_or_expr, uexpr, vars_)
    if not rec.recognized:
        raise NotImplementedError(
            "PDE is not recognized as generalized Clairaut complete-integral form."
        )
    if parameter_symbols is None:
        params = tuple(sp.Symbol(f"C{i + 1}", real=True) for i in range(len(vars_)))
    else:
        params = tuple(parameter_symbols)
        if len(params) != len(vars_):
            raise ValueError(
                "parameter_symbols must have one symbol per independent variable."
            )
    subs = dict(zip(rec.gradients, params))
    rhs = sp.expand(sum(v * a for v, a in zip(vars_, params)) + rec.phi.xreplace(subs))
    sol = sp.Eq(uexpr, rhs)
    singular = detect_singular_solution_from_complete_integral(sol, params, uexpr)
    details = {"recognition": rec, "parameters": params}
    if singular is not None:
        details["singular_solution"] = singular
        details["envelope_solution"] = singular
    return _complete_integral_result(
        "generalized_clairaut_complete_integral",
        [sol],
        details,
        eq_or_expr,
        uexpr,
        vars_,
    )


def construct_envelope_from_complete_integral(solution_eq, parameter_symbols, dep_expr):
    if not isinstance(solution_eq, sp.Equality) or solution_eq.lhs != dep_expr:
        return None
    rhs = sp.expand(solution_eq.rhs)
    eqs = [sp.Eq(sp.diff(rhs, a), 0) for a in parameter_symbols]
    try:
        sols = sp.solve(eqs, parameter_symbols, dict=True)
    except Exception:
        sols = []
    out = []
    for sol in sols:
        try:
            out.append(sp.Eq(dep_expr, sp.simplify(rhs.subs(sol))))
        except Exception:
            continue
    return tuple(dict.fromkeys(out)) or None


def detect_singular_solution_from_complete_integral(
    solution_eq, parameter_symbols, dep_expr
):
    env = construct_envelope_from_complete_integral(
        solution_eq, parameter_symbols, dep_expr
    )
    if not env:
        return None
    return env[0] if len(env) == 1 else env


def _first_order_nonlinear_data(eq_or_expr, dep_expr_or_func, indep_vars=None):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    zero = _as_zero_expr(eq_or_expr)
    vars_ = tuple(vars_)
    if len(vars_) < 2:
        raise ValueError("Need at least two independent variables.")
    for node in sp.preorder_traversal(zero):
        if isinstance(node, sp.Derivative) and node.expr == uexpr:
            total_order = sum(c for _, c in node.variable_count)
            if total_order > 1:
                raise ValueError("PDE is not first-order.")
    grads = [sp.Symbol(f"P{i}") for i in range(len(vars_))]
    U0 = sp.Symbol("U0")
    reps = {uexpr: U0}
    for i, v in enumerate(vars_):
        reps[sp.diff(uexpr, v)] = grads[i]
    F = sp.expand(zero.xreplace(reps))
    return uexpr, vars_, U0, tuple(grads), F


def _verify_first_order_solution(eq_or_expr, solution_eq, dep_expr, vars_):
    """
    Verify an explicit or mildly implicit first-order solution candidate.

    Supported verification modes:
      - explicit equality dep_expr == rhs,
      - implicit equalities H(dep_expr, x, y, ...) == const, when SymPy can
        solve for dep_expr on at least one explicit branch.

    Returns
    -------
    (ok, residual)
      ok is False when verification is unavailable or fails.
      residual is the substituted PDE residual when available.
    """
    explicit_candidates = []
    if isinstance(solution_eq, sp.Equality):
        if solution_eq.lhs == dep_expr:
            explicit_candidates = [sp.expand(solution_eq.rhs)]
        else:
            try:
                branches = sp.solve(solution_eq, dep_expr, dict=False)
            except Exception:
                branches = []
            if branches is None:
                branches = []
            if not isinstance(branches, (list, tuple)):
                branches = [branches]
            explicit_candidates = [sp.expand(b) for b in branches if b is not None]

    for rhs in explicit_candidates:
        try:
            residual = sp.expand(_as_zero_expr(eq_or_expr).subs({dep_expr: rhs}).doit())
        except Exception:
            continue
        try:
            ok = sp.simplify(residual) == 0
        except Exception:
            ok = False
        if ok:
            return True, residual

    return False, None


def _pfaffian_exactness_matrix(p_fields, vars_):
    return [
        [
            sp.simplify(sp.diff(p_fields[i], vars_[j]) - sp.diff(p_fields[j], vars_[i]))
            for j in range(len(vars_))
        ]
        for i in range(len(vars_))
    ]


def _pfaffian_is_exact(p_fields, vars_):
    mat = _pfaffian_exactness_matrix(p_fields, vars_)
    return all(
        mat[i][j] == 0 for i in range(len(vars_)) for j in range(i + 1, len(vars_))
    )


def _integrate_pfaffian_in_order(p_fields, vars_, order):
    expr = sp.integrate(p_fields[order[0]], vars_[order[0]])
    for idx in order[1:]:
        mismatch = sp.simplify(p_fields[idx] - sp.diff(expr, vars_[idx]))
        if mismatch != 0:
            # restricted robustness: only accept corrections depending on the current variable
            other = set(vars_) - {vars_[idx]}
            if not mismatch.free_symbols.isdisjoint(other):
                raise ValueError(
                    "Pfaffian correction depends on too many variables for restricted integration."
                )
            expr = sp.expand(expr + sp.integrate(mismatch, vars_[idx]))
    return sp.expand(expr)


def _one_form_is_exact(components, vars_):
    for i in range(len(vars_)):
        for j in range(i + 1, len(vars_)):
            if (
                sp.simplify(
                    sp.diff(components[i], vars_[j]) - sp.diff(components[j], vars_[i])
                )
                != 0
            ):
                return False
    return True


def _integrate_exact_one_form(components, vars_):
    from itertools import permutations

    candidates = []
    n = len(vars_)
    for order in permutations(range(n)):
        try:
            H = sp.integrate(components[order[0]], vars_[order[0]])
            used = {vars_[order[0]]}
            ok = True
            for idx in order[1:]:
                mismatch = sp.simplify(components[idx] - sp.diff(H, vars_[idx]))
                if mismatch != 0 and not mismatch.free_symbols.isdisjoint(used):
                    ok = False
                    break
                H = sp.expand(H + sp.integrate(mismatch, vars_[idx]))
                used.add(vars_[idx])
            if ok and all(
                sp.simplify(sp.diff(H, vars_[i]) - components[i]) == 0 for i in range(n)
            ):
                candidates.append(sp.expand(H))
        except Exception:
            pass
    if not candidates:
        raise ValueError(
            "Could not integrate exact Pfaffian one-form in the restricted solver."
        )
    return min(candidates, key=_expr_complexity)


def _build_implicit_solution_from_potential(
    H, dep_expr, dep_symbol, *, constant_symbol=None
):
    c0 = constant_symbol if constant_symbol is not None else sp.Symbol("C0", real=True)
    implicit_eq = sp.Eq(sp.expand(H.subs(dep_symbol, dep_expr)), c0)
    try:
        branches = sp.solve(implicit_eq, dep_expr, dict=False)
    except Exception:
        branches = []
    if branches is None:
        branches = []
    if not isinstance(branches, (list, tuple)):
        branches = [branches]
    branches = [sp.expand(b) for b in branches if b is not None]
    if branches:
        best = min(branches, key=_expr_complexity)
        return sp.Eq(dep_expr, best)
    return implicit_eq


def _simple_integrating_factor_pfaffian_2vars(p_fields, vars_, dep_symbol):
    """
    Restricted integrating-factor detection for the Pfaffian one-form

        du - p(x,y,u) dx - q(x,y,u) dy = 0.

    Tries integrating factors depending only on x, only on y, or only on u.
    Returns transformed one-form components (M, N, P) when successful.
    """
    if len(vars_) != 2:
        return None
    x, y = vars_
    U = dep_symbol
    p = sp.expand(sp.sympify(p_fields[0]))
    q = sp.expand(sp.sympify(p_fields[1]))
    M, N, _ = -p, -q, sp.Integer(1)

    def _factor_from_log_derivative(logd, var):
        try:
            return sp.exp(sp.integrate(logd, var))
        except Exception:
            return None

    # mu(x)
    f = sp.simplify(sp.diff(M, U))
    if f != 0 and f.free_symbols.issubset({x}) and sp.simplify(sp.diff(N, U)) == 0:
        if sp.simplify(sp.diff(M, y) - sp.diff(N, x) - f * N) == 0:
            mu = _factor_from_log_derivative(f, x)
            if mu is not None:
                return [sp.expand(mu * M), sp.expand(mu * N), sp.expand(mu)]

    # mu(y)
    g = sp.simplify(sp.diff(N, U))
    if g != 0 and g.free_symbols.issubset({y}) and sp.simplify(sp.diff(M, U)) == 0:
        if sp.simplify(sp.diff(N, x) - sp.diff(M, y) - g * M) == 0:
            mu = _factor_from_log_derivative(g, y)
            if mu is not None:
                return [sp.expand(mu * M), sp.expand(mu * N), sp.expand(mu)]

    # mu(u)
    if sp.simplify(sp.diff(M, y) - sp.diff(N, x)) == 0:
        ratios = []
        for comp in (M, N):
            if sp.simplify(comp) == 0:
                continue
            ratio = sp.simplify(-sp.diff(comp, U) / comp)
            if ratio.free_symbols.issubset({U}):
                ratios.append(ratio)
            else:
                ratios = None
                break
        if ratios is not None:
            if not ratios:
                ratio = sp.Integer(0)
            else:
                ratio = ratios[0]
                if all(sp.simplify(r - ratio) == 0 for r in ratios[1:]):
                    pass
                else:
                    ratio = None
            if ratio is not None:
                mu = _factor_from_log_derivative(ratio, U)
                if mu is not None:
                    return [sp.expand(mu * M), sp.expand(mu * N), sp.expand(mu)]

    return None


def integrate_pfaffian_equation(
    p_fields,
    dep_expr_or_func,
    indep_vars=None,
    *,
    dependent_symbol=None,
    allow_implicit=True,
    constant_symbol=None,
):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    vars_ = tuple(vars_)
    p_fields = [sp.expand(sp.sympify(p)) for p in p_fields]
    if len(p_fields) != len(vars_):
        raise ValueError("Need one derivative field per independent variable.")

    dep_symbol = dependent_symbol if dependent_symbol is not None else None
    dep_present = dep_symbol is not None and any(
        dep_symbol in pf.free_symbols for pf in p_fields
    )

    # Exact explicit-gradient case first, but only when the derivative fields do
    # not depend on the dependent variable surrogate.
    if not dep_present and _pfaffian_is_exact(p_fields, vars_):
        candidates = []
        orders = [tuple(range(len(vars_)))]
        if len(vars_) == 2:
            orders.append((1, 0))
        for order in orders:
            try:
                candidates.append(_integrate_pfaffian_in_order(p_fields, vars_, order))
            except Exception:
                pass
        if candidates:
            expr = min(candidates, key=_expr_complexity)
            return sp.Eq(uexpr, expr)

    # Restricted additive fallback, likewise only in the explicit-gradient case.
    if (not dep_present) and all(
        p_fields[i].free_symbols.isdisjoint(set(vars_) - {vars_[i]})
        for i in range(len(vars_))
    ):
        expr = sp.expand(
            sum(sp.integrate(p_fields[i], vars_[i]) for i in range(len(vars_)))
        )
        return sp.Eq(uexpr, expr)

    # Use the Pfaffian one-form du - p dx - q dy = 0 for the two-variable case.
    if len(vars_) == 2 and allow_implicit:
        x, y = vars_
        dep_symbol = (
            dependent_symbol if dependent_symbol is not None else sp.Symbol("U_pf")
        )
        subs_dep = {uexpr: dep_symbol}
        M = sp.expand(-p_fields[0].xreplace(subs_dep))
        N = sp.expand(-p_fields[1].xreplace(subs_dep))
        P = sp.Integer(1)
        comps = [M, N, P]
        ext_vars = (x, y, dep_symbol)

        if _one_form_is_exact(comps, ext_vars):
            H = _integrate_exact_one_form(comps, ext_vars)
            return _build_implicit_solution_from_potential(
                H, uexpr, dep_symbol, constant_symbol=constant_symbol
            )

        transformed = _simple_integrating_factor_pfaffian_2vars(
            p_fields, vars_, dep_symbol
        )
        if transformed is not None and _one_form_is_exact(transformed, ext_vars):
            H = _integrate_exact_one_form(transformed, ext_vars)
            return _build_implicit_solution_from_potential(
                H, uexpr, dep_symbol, constant_symbol=constant_symbol
            )

    raise ValueError("Pfaffian system is not obviously exact in the restricted solver.")


def _verification_entry_first_order(eq_or_expr, solution_eq, dep_expr, vars_):
    ok, residual = _verify_first_order_solution(
        eq_or_expr, solution_eq, dep_expr, vars_
    )
    return {
        "solution": solution_eq,
        "verified": bool(ok),
        "residual": residual,
    }


def _complete_integral_result(method, solutions, details, eq_or_expr, dep_expr, vars_):
    uniq = []
    seen = set()
    for sol in solutions:
        sig = sp.srepr(sp.expand(sol.rhs))
        if sig not in seen:
            seen.add(sig)
            uniq.append(sol)
    verification = tuple(
        _verification_entry_first_order(eq_or_expr, sol, dep_expr, vars_)
        for sol in uniq
    )
    return CompleteIntegralResult(method, tuple(uniq), dict(details), verification)


def _charpit_auxiliary_system(F, U0, x, y, p, q):
    Fx = sp.expand(sp.diff(F, x))
    Fy = sp.expand(sp.diff(F, y))
    Fu = sp.expand(sp.diff(F, U0))
    Fp = sp.expand(sp.diff(F, p))
    Fq = sp.expand(sp.diff(F, q))
    return {
        "Fx": Fx,
        "Fy": Fy,
        "Fu": Fu,
        "Fp": Fp,
        "Fq": Fq,
        "dx_ds": -Fp,
        "dy_ds": -Fq,
        "du_ds": -(p * Fp + q * Fq),
        "dp_ds": Fx + p * Fu,
        "dq_ds": Fy + q * Fu,
    }


def _safe_ratio(num, den):
    den = sp.simplify(den)
    if den in (sp.S.Zero, 0) or den is sp.zoo:
        return None
    try:
        out = sp.simplify(sp.together(num / den))
    except Exception:
        try:
            out = sp.expand(num / den)
        except Exception:
            return None
    if out in (sp.zoo, sp.oo, -sp.oo) or out.has(sp.zoo):
        return None
    return out


def _solve_scalar_first_order_ode(rhs, dep_symbol, indep_symbol):
    Y = sp.Function(f"{dep_symbol.name}_{indep_symbol.name}")
    eq = sp.Eq(
        sp.diff(Y(indep_symbol), indep_symbol), rhs.subs(dep_symbol, Y(indep_symbol))
    )
    try:
        sol = sp.dsolve(eq)
    except Exception:
        sol = None
    if isinstance(sol, sp.Equality):
        return sp.expand(sol.rhs)
    return None


def _charpit_try_dpdx(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr):
    rhs = _safe_ratio(aux["dp_ds"], aux["dx_ds"])
    if rhs is None or not rhs.free_symbols.isdisjoint({y, U0, q}):
        return []
    pexpr = _solve_scalar_first_order_ode(rhs, p, x)
    if pexpr is None:
        return []
    try:
        qsols = sp.solve(sp.Eq(F.subs({p: pexpr}), 0), q, dict=False)
    except Exception:
        qsols = []
    sols = []
    qsols = list(qsols) if isinstance(qsols, (list, tuple)) else [qsols]
    for qexpr in qsols:
        if qexpr is None:
            continue
        try:
            cand = integrate_pfaffian_equation(
                [pexpr, sp.expand(qexpr)], uexpr, vars_, dependent_symbol=U0
            )
        except Exception:
            continue
        ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
        if ok:
            sols.append(cand)
    return sols


def _charpit_try_dqdy(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr):
    rhs = _safe_ratio(aux["dq_ds"], aux["dy_ds"])
    if rhs is None or not rhs.free_symbols.isdisjoint({x, U0, p}):
        return []
    qexpr = _solve_scalar_first_order_ode(rhs, q, y)
    if qexpr is None:
        return []
    try:
        psols = sp.solve(sp.Eq(F.subs({q: qexpr}), 0), p, dict=False)
    except Exception:
        psols = []
    sols = []
    psols = list(psols) if isinstance(psols, (list, tuple)) else [psols]
    for pexpr in psols:
        if pexpr is None:
            continue
        try:
            cand = integrate_pfaffian_equation(
                [sp.expand(pexpr), qexpr], uexpr, vars_, dependent_symbol=U0
            )
        except Exception:
            continue
        ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
        if ok:
            sols.append(cand)
    return sols


def _charpit_try_dqdx(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr):
    rhs = _safe_ratio(aux["dq_ds"], aux["dx_ds"])
    if rhs is None or not rhs.free_symbols.isdisjoint({y, U0, p}):
        return []
    qexpr = _solve_scalar_first_order_ode(rhs, q, x)
    if qexpr is None:
        return []
    try:
        psols = sp.solve(sp.Eq(F.subs({q: qexpr}), 0), p, dict=False)
    except Exception:
        psols = []
    sols = []
    psols = list(psols) if isinstance(psols, (list, tuple)) else [psols]
    for pexpr in psols:
        if pexpr is None:
            continue
        try:
            cand = integrate_pfaffian_equation(
                [sp.expand(pexpr), sp.expand(qexpr)], uexpr, vars_, dependent_symbol=U0
            )
        except Exception:
            continue
        ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
        if ok:
            sols.append(cand)
    return sols


def _charpit_try_dpdy(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr):
    rhs = _safe_ratio(aux["dp_ds"], aux["dy_ds"])
    if rhs is None or not rhs.free_symbols.isdisjoint({x, U0, q}):
        return []
    pexpr = _solve_scalar_first_order_ode(rhs, p, y)
    if pexpr is None:
        return []
    try:
        qsols = sp.solve(sp.Eq(F.subs({p: pexpr}), 0), q, dict=False)
    except Exception:
        qsols = []
    sols = []
    qsols = list(qsols) if isinstance(qsols, (list, tuple)) else [qsols]
    for qexpr in qsols:
        if qexpr is None:
            continue
        try:
            cand = integrate_pfaffian_equation(
                [sp.expand(pexpr), sp.expand(qexpr)], uexpr, vars_, dependent_symbol=U0
            )
        except Exception:
            continue
        ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
        if ok:
            sols.append(cand)
    return sols


def _charpit_try_dpdq_autonomous(F, aux, p, q, x, y, uexpr, vars_, eq_or_expr):
    rhs = _safe_ratio(aux["dp_ds"], aux["dq_ds"])
    if rhs is None or not rhs.free_symbols.isdisjoint({x, y, uexpr}):
        return []
    Pq = sp.Function("Pq")
    try:
        psol = sp.dsolve(sp.Eq(sp.diff(Pq(q), q), rhs.subs(p, Pq(q))))
    except Exception:
        psol = None
    if not isinstance(psol, sp.Equality):
        return []
    a = sp.Symbol("a", real=True)
    relation = sp.expand(psol.rhs.subs(q, a))
    # Semi-automatic elimination: keep either explicit p(a) or implicit relation R(p,a)=0.
    p_candidates = []
    try:
        p_candidates = sp.solve(sp.Eq(p, relation), p, dict=False)
    except Exception:
        p_candidates = []
    if not p_candidates:
        p_candidates = [relation]
    sols = []
    for pval in (
        p_candidates if isinstance(p_candidates, (list, tuple)) else [p_candidates]
    ):
        if pval is None:
            continue
        try:
            qvals = sp.solve(sp.Eq(F.subs({p: pval}), 0), q, dict=False)
        except Exception:
            qvals = []
        for qval in qvals if isinstance(qvals, (list, tuple)) else [qvals]:
            if qval is None:
                continue
            sol_expr = sp.expand(
                sp.sympify(pval) * x + sp.sympify(qval) * y + sp.Symbol("C0", real=True)
            )
            cand = sp.Eq(uexpr, sol_expr)
            ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
            if ok:
                sols.append(cand)
    return sols


def _charpit_try_dpdu(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr):
    rhs = _safe_ratio(aux["dp_ds"], aux["du_ds"])
    if rhs is None or not rhs.free_symbols.isdisjoint({x, y, q}):
        return []
    pU = sp.Function("pU")
    try:
        psol = sp.dsolve(sp.Eq(sp.diff(pU(U0), U0), rhs.subs(p, pU(U0))))
    except Exception:
        psol = None
    if not isinstance(psol, sp.Equality):
        return []
    pexpr = sp.expand(psol.rhs)
    try:
        qsols = sp.solve(sp.Eq(F.subs({p: pexpr}), 0), q, dict=False)
    except Exception:
        qsols = []
    sols = []
    qsols = list(qsols) if isinstance(qsols, (list, tuple)) else [qsols]
    for qexpr in qsols:
        if qexpr is None:
            continue
        try:
            cand = integrate_pfaffian_equation(
                [sp.expand(pexpr), sp.expand(qexpr)], uexpr, vars_, dependent_symbol=U0
            )
        except Exception:
            continue
        ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
        if ok:
            sols.append(cand)
    return sols


def _charpit_try_dqdu(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr):
    rhs = _safe_ratio(aux["dq_ds"], aux["du_ds"])
    if rhs is None or not rhs.free_symbols.isdisjoint({x, y, p}):
        return []
    qU = sp.Function("qU")
    try:
        qsol = sp.dsolve(sp.Eq(sp.diff(qU(U0), U0), rhs.subs(q, qU(U0))))
    except Exception:
        qsol = None
    if not isinstance(qsol, sp.Equality):
        return []
    qexpr = sp.expand(qsol.rhs)
    try:
        psols = sp.solve(sp.Eq(F.subs({q: qexpr}), 0), p, dict=False)
    except Exception:
        psols = []
    sols = []
    psols = list(psols) if isinstance(psols, (list, tuple)) else [psols]
    for pexpr in psols:
        if pexpr is None:
            continue
        try:
            cand = integrate_pfaffian_equation(
                [sp.expand(pexpr), sp.expand(qexpr)], uexpr, vars_, dependent_symbol=U0
            )
        except Exception:
            continue
        ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
        if ok:
            sols.append(cand)
    return sols


def _charpit_try_dxdy_safe(
    F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr, parameter_symbol
):
    """
    Safe characteristic dx/dy branch for homogeneous first-order linear/quasilinear
    forms A(x,y) p + B(x,y) q = 0 with u constant on characteristics.
    Returns complete-integral-style linear representatives u = a*C(x,y) + C0.
    """
    A = sp.expand(sp.diff(F, p))
    B = sp.expand(sp.diff(F, q))
    rem = sp.expand(F - A * p - B * q)
    if rem != 0:
        return []
    if not A.free_symbols.issubset({x, y}) or not B.free_symbols.issubset({x, y}):
        return []
    rhs = _safe_ratio(aux["dx_ds"], aux["dy_ds"])
    if rhs is None or not rhs.free_symbols.issubset({x, y}):
        return []

    Y = sp.Function("XofY")
    invariant = None
    try:
        sol = sp.dsolve(sp.Eq(sp.diff(Y(y), y), rhs.subs(x, Y(y))))
    except Exception:
        sol = None
    if isinstance(sol, sp.Equality):
        expr = sp.expand(sol.lhs - sol.rhs)
        if not expr.has(Y(y)):
            invariant = expr

    if invariant is None:
        # fallback via dy/dx
        rhs2 = _safe_ratio(aux["dy_ds"], aux["dx_ds"])
        if rhs2 is None or not rhs2.free_symbols.issubset({x, y}):
            return []
        Z = sp.Function("YofX")
        try:
            sol2 = sp.dsolve(sp.Eq(sp.diff(Z(x), x), rhs2.subs(y, Z(x))))
        except Exception:
            sol2 = None
        if isinstance(sol2, sp.Equality):
            expr2 = sp.expand(sol2.lhs - sol2.rhs)
            if not expr2.has(Z(x)):
                invariant = expr2
    if invariant is None:
        return []

    a = parameter_symbol if parameter_symbol is not None else sp.Symbol("a", real=True)
    c0 = sp.Symbol("C0", real=True)
    cand = sp.Eq(uexpr, sp.expand(a * invariant + c0))
    ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
    return [cand] if ok else []


def _charpit_separate_additive(F, x, y, p, q):
    expr = sp.expand(F)
    terms = list(expr.args) if isinstance(expr, sp.Add) else [expr]
    left_terms = []
    right_terms = []
    for term in terms:
        fs = term.free_symbols
        # x-p side allows only x and p
        if fs.issubset({x, p}):
            left_terms.append(term)
        elif fs.issubset({y, q}):
            right_terms.append(term)
        elif p not in fs and x not in fs:
            right_terms.append(term)
        elif q not in fs and y not in fs:
            left_terms.append(term)
        else:
            return None
    if not left_terms or not right_terms:
        return None
    return sp.expand(sum(left_terms)), sp.expand(sum(right_terms))


def _charpit_try_separated(F, x, y, U0, p, q, uexpr, vars_, eq_or_expr):
    sep = _charpit_separate_additive(F, x, y, p, q)
    if sep is None:
        return []
    Aexpr, Bexpr = sep
    a = sp.Symbol("a", real=True)
    try:
        psols = sp.solve(sp.Eq(Aexpr, a), p, dict=False)
        qsols = sp.solve(sp.Eq(Bexpr, -a), q, dict=False)
    except Exception:
        return []
    sols = []
    psols = list(psols) if isinstance(psols, (list, tuple)) else [psols]
    qsols = list(qsols) if isinstance(qsols, (list, tuple)) else [qsols]
    for pexpr in psols:
        for qexpr in qsols:
            if pexpr is None or qexpr is None:
                continue
            try:
                cand = integrate_pfaffian_equation(
                    [sp.expand(pexpr), sp.expand(qexpr)],
                    uexpr,
                    vars_,
                    dependent_symbol=U0,
                )
            except Exception:
                continue
            ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
            if ok:
                sols.append(cand)
    return sols


def solve_charpit_complete_integral_2vars(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, parameter_symbol=None
):
    """
    Restricted Charpit-style complete-integral solver for two-variable
    first-order nonlinear PDEs F(x,y,u,p,q)=0.

    Implemented strategy:
      1. build Charpit auxiliary equations,
      2. search tractable ratios dp/dx, dq/dy, dp/dq,
      3. solve tractable auxiliary ODEs,
      4. reconstruct u via Pfaffian integration where possible,
      5. verify each candidate solution.

    Supported strongest cases:
      - autonomous F(p,q)=0 complete-integral families,
      - additive separated forms A(x,p)+B(y,q)=0,
      - tractable auxiliary ODEs with univariate rhs.
    """
    uexpr, vars_, U0, grads, F = _first_order_nonlinear_data(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    if len(vars_) != 2:
        raise ValueError("Restricted Charpit solver is for two variables.")
    x, y = vars_
    p, q = grads
    a = parameter_symbol if parameter_symbol is not None else sp.Symbol("a", real=True)
    c0 = sp.Symbol("C0", real=True)

    sols = []

    # Case 1: autonomous complete integral u = a x + q(a) y + C0.
    if F.free_symbols.isdisjoint({x, y, U0}):
        try:
            qsols = sp.solve(sp.Eq(F.subs({p: a}), 0), q, dict=False)
        except Exception:
            qsols = []
        for qexpr in qsols if isinstance(qsols, (list, tuple)) else [qsols]:
            if qexpr is None:
                continue
            cand = sp.Eq(uexpr, sp.expand(a * x + sp.sympify(qexpr) * y + c0))
            ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
            if ok:
                sols.append(cand)
        if sols:
            aux = _charpit_auxiliary_system(F, U0, x, y, p, q)
            return _complete_integral_result(
                "charpit_autonomous_complete_integral",
                sols,
                {"F": F, "parameter": a, "auxiliary_system": aux},
                eq_or_expr,
                uexpr,
                vars_,
            )

    aux = _charpit_auxiliary_system(F, U0, x, y, p, q)

    # Case 2: additive separation.
    sols = _charpit_try_separated(F, x, y, U0, p, q, uexpr, vars_, eq_or_expr)
    if sols:
        return _complete_integral_result(
            "charpit_additively_separated",
            sols,
            {"F": F, "auxiliary_system": aux},
            eq_or_expr,
            uexpr,
            vars_,
        )

    # Case 3: tractable auxiliary ratios.
    sols = _charpit_try_dpdx(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr)
    if sols:
        return _complete_integral_result(
            "charpit_dpdx_reconstruction",
            sols,
            {"F": F, "auxiliary_system": aux, "searched_ratios": ("dp/dx",)},
            eq_or_expr,
            uexpr,
            vars_,
        )

    sols = _charpit_try_dqdy(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr)
    if sols:
        return _complete_integral_result(
            "charpit_dqdy_reconstruction",
            sols,
            {"F": F, "auxiliary_system": aux, "searched_ratios": ("dq/dy",)},
            eq_or_expr,
            uexpr,
            vars_,
        )

    sols = _charpit_try_dpdu(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr)
    if sols:
        return _complete_integral_result(
            "charpit_dpdu_reconstruction",
            sols,
            {"F": F, "auxiliary_system": aux, "searched_ratios": ("dp/du",)},
            eq_or_expr,
            uexpr,
            vars_,
        )

    sols = _charpit_try_dqdu(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr)
    if sols:
        return _complete_integral_result(
            "charpit_dqdu_reconstruction",
            sols,
            {"F": F, "auxiliary_system": aux, "searched_ratios": ("dq/du",)},
            eq_or_expr,
            uexpr,
            vars_,
        )

    sols = _charpit_try_dxdy_safe(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr, a)
    if sols:
        return _complete_integral_result(
            "charpit_dxdy_safe_complete_integral",
            sols,
            {"F": F, "auxiliary_system": aux, "searched_ratios": ("dx/dy",)},
            eq_or_expr,
            uexpr,
            vars_,
        )

    sols = _charpit_try_dqdx(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr)
    if sols:
        return _complete_integral_result(
            "charpit_dqdx_reconstruction",
            sols,
            {"F": F, "auxiliary_system": aux, "searched_ratios": ("dq/dx",)},
            eq_or_expr,
            uexpr,
            vars_,
        )

    sols = _charpit_try_dpdy(F, aux, x, y, U0, p, q, uexpr, vars_, eq_or_expr)
    if sols:
        return _complete_integral_result(
            "charpit_dpdy_reconstruction",
            sols,
            {"F": F, "auxiliary_system": aux, "searched_ratios": ("dp/dy",)},
            eq_or_expr,
            uexpr,
            vars_,
        )

    sols = _charpit_try_dpdq_autonomous(F, aux, p, q, x, y, uexpr, vars_, eq_or_expr)
    if sols:
        return _complete_integral_result(
            "charpit_dpdq_complete_integral",
            sols,
            {"F": F, "auxiliary_system": aux, "searched_ratios": ("dp/dq",)},
            eq_or_expr,
            uexpr,
            vars_,
        )

    raise NotImplementedError(
        "Restricted Charpit solver could not find a tractable auxiliary equation or separated form."
    )


def _jacobi_try_additive_separation(F, vars_, grads, params):
    expr = sp.expand(F)
    terms = list(expr.args) if isinstance(expr, sp.Add) else [expr]
    groups = {i: [] for i in range(len(vars_))}
    for term in terms:
        placed = False
        fs = term.free_symbols
        for i, (v, gi) in enumerate(zip(vars_, grads)):
            if fs.issubset({v, gi}) or (
                gi in fs and all((sym in {v, gi}) for sym in fs)
            ):
                groups[i].append(term)
                placed = True
                break
        if not placed:
            return None
    if any(len(v) == 0 for v in groups.values()):
        return None
    return [sp.expand(sum(groups[i])) for i in range(len(vars_))]


def solve_jacobi_complete_integral(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, parameter_symbols=None
):
    """
    Restricted Jacobi-style complete-integral solver for first-order PDEs in
    3 or more variables.

    Supported cases:
      - autonomous gradient-only PDEs F(p1,...,pn)=0,
      - partially separable additive forms sum_i G_i(x_i, p_i)=0.
    """
    uexpr, vars_, U0, grads, F = _first_order_nonlinear_data(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    n = len(vars_)
    if n < 3:
        raise ValueError("Restricted Jacobi solver is for 3 or more variables.")

    if parameter_symbols is None:
        params = sp.symbols("a0:%d" % max(n - 1, 1), real=True)
    else:
        params = tuple(parameter_symbols)
        if len(params) != n - 1:
            raise ValueError("Need n-1 parameter symbols.")
    c0 = sp.Symbol("C0", real=True)

    # Case 1: additive separation across (x_i, p_i) pairs.
    if U0 not in F.free_symbols:
        sep_groups = _jacobi_try_additive_separation(F, vars_, grads, params)
        if sep_groups is not None:
            sols = []
            target_values = list(params) + [-sum(params)]
            gradient_exprs = []
            ok_all = True
            for i, grp in enumerate(sep_groups):
                try:
                    roots = sp.solve(sp.Eq(grp, target_values[i]), grads[i], dict=False)
                except Exception:
                    roots = []
                roots = list(roots) if isinstance(roots, (list, tuple)) else [roots]
                if not roots or roots[0] is None:
                    ok_all = False
                    break
                gradient_exprs.append(sp.expand(roots[0]))
            if ok_all:
                try:
                    cand = integrate_pfaffian_equation(gradient_exprs, uexpr, vars_)
                except Exception:
                    cand = None
                if cand is not None:
                    ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
                    if ok:
                        sols.append(cand)
                if sols:
                    return _complete_integral_result(
                        "jacobi_additively_separated_complete_integral",
                        sols,
                        {"F": F, "parameters": params},
                        eq_or_expr,
                        uexpr,
                        vars_,
                    )

    # Case 2: autonomous or partially separable gradient-only affine complete integral.
    if F.free_symbols.isdisjoint(set(vars_) | {U0}):
        subs = {grads[i]: params[i] for i in range(n - 1)}
        target = grads[-1]
        try:
            roots = sp.solve(sp.Eq(F.subs(subs), 0), target, dict=False)
        except Exception:
            roots = []

        sols = []
        for root in roots if isinstance(roots, (list, tuple)) else [roots]:
            if root is None:
                continue
            sol_expr = sp.expand(
                sum(params[i] * vars_[i] for i in range(n - 1))
                + sp.sympify(root) * vars_[-1]
                + c0
            )
            cand = sp.Eq(uexpr, sol_expr)
            ok, _ = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
            if ok:
                sols.append(cand)

        if sols:
            return _complete_integral_result(
                "jacobi_autonomous_complete_integral",
                sols,
                {"F": F, "parameters": params},
                eq_or_expr,
                uexpr,
                vars_,
            )

    raise NotImplementedError(
        "Restricted Jacobi solver could not build a compatible complete integral."
    )


def solve_complete_integral_pde(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True, **kwargs
):
    """Dispatch restricted complete-integral methods and return a structured result."""
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    zero = _as_zero_expr(eq_or_expr)
    order = 0
    for node in sp.preorder_traversal(zero):
        if isinstance(node, sp.Derivative) and node.expr == uexpr:
            order = max(order, sum(c for _, c in node.variable_count))
    if order != 1:
        raise ValueError(
            "Complete-integral methods are only implemented for first-order PDEs."
        )
    if len(vars_) == 2:
        return solve_charpit_complete_integral_2vars(eq_or_expr, uexpr, vars_, **kwargs)
    return solve_jacobi_complete_integral(eq_or_expr, uexpr, vars_, **kwargs)


@dataclass(frozen=True)
class InitialCurve2D:
    parameter: sp.Symbol
    x_curve: sp.Expr
    y_curve: sp.Expr
    u_data: sp.Expr
    source: object | None = None


@dataclass(frozen=True)
class FirstOrderCauchyProblemResult:
    method: str
    solution: sp.Equality
    initial_curve: InitialCurve2D
    details: dict
    verification: dict


def process_initial_curve_2d(
    initial, dep_expr_or_func, indep_vars=None, *, parameter=None
):
    """
    Normalize a first-order initial condition into a parametric initial curve.

    Supported restricted forms:
      - Eq(u(x, phi(x)), psi(x))
      - Eq(u(phi(y), y), psi(y))
      - Eq(u(x0(s), y0(s)), u0(s)) where `parameter` is given and the
        expressions depend only on that parameter.
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if tuple(vars_)[:2] != tuple(vars_) or len(vars_) != 2:
        if len(vars_) != 2:
            raise ValueError(
                "Initial-curve processing is only implemented for two variables."
            )
    x, y = vars_
    if not isinstance(initial, sp.Equality):
        raise ValueError("Initial data must be an Equality.")
    lhs, rhs = initial.lhs, initial.rhs
    if not (
        isinstance(lhs, AppliedUndef) and lhs.func == uexpr.func and len(lhs.args) == 2
    ):
        raise ValueError("Initial condition must be of the form Eq(u(...,...), data).")
    a, b = lhs.args

    # Explicit parameter provided: require all curve/data expressions depend only on it.
    if parameter is not None:
        s = (
            sp.Symbol(str(parameter), real=True)
            if not isinstance(parameter, sp.Symbol)
            else parameter
        )
        if (
            (a.free_symbols - {s})
            or (b.free_symbols - {s})
            or (sp.sympify(rhs).free_symbols - {s})
        ):
            raise ValueError(
                "Parametric initial curve must depend only on the supplied parameter."
            )
        return InitialCurve2D(
            s, sp.expand(a), sp.expand(b), sp.expand(rhs), source=initial
        )

    # Normalize the one-form before exactness and integrating-factor checks to keep equivalent Pfaffian forms comparable.
    if a == x and rhs.free_symbols.isdisjoint({y}):
        return InitialCurve2D(
            x, sp.expand(x), sp.expand(b), sp.expand(rhs), source=initial
        )
    if b == y and rhs.free_symbols.isdisjoint({x}):
        return InitialCurve2D(
            y, sp.expand(a), sp.expand(y), sp.expand(rhs), source=initial
        )

    # Fallback: if exactly one variable parametrizes all pieces.
    free = (a.free_symbols | b.free_symbols | sp.sympify(rhs).free_symbols) & {x, y}
    if len(free) == 1:
        s = list(free)[0]
        return InitialCurve2D(
            s, sp.expand(a), sp.expand(b), sp.expand(rhs), source=initial
        )

    raise NotImplementedError(
        "Initial curve is outside the restricted supported forms."
    )


def _free_function_atoms(expr):
    out = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, AppliedUndef):
            out.append(node)
    uniq = []
    for node in out:
        if node not in uniq:
            uniq.append(node)
    return uniq


def _fit_single_free_function_family_2d(
    solution_eq, curve: InitialCurve2D, dep_expr_or_func, indep_vars=None
):
    """
    Fit a family u(x,y) = F(g(x,y)) + h(x,y) to initial-curve data by solving
    for the one arbitrary single-argument function F.
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    rhs = sp.expand(solution_eq.rhs)
    funs = [f for f in _free_function_atoms(rhs) if len(f.args) == 1]
    if len(funs) != 1:
        return None
    atom = funs[0]
    Fsym = atom.func
    arg_expr = sp.expand(atom.args[0])
    x, y = vars_
    s = curve.parameter
    arg_on_curve = sp.expand(arg_expr.subs({x: curve.x_curve, y: curve.y_curve}))
    rhs_on_curve = sp.expand(rhs.subs({x: curve.x_curve, y: curve.y_curve}))
    # Replace occurrences of the same arbitrary function on the initial curve by 0 to isolate the non-free part.
    base_on_curve = sp.expand(
        rhs_on_curve.replace(
            lambda e: isinstance(e, AppliedUndef) and e.func == Fsym,
            lambda e: sp.Integer(0),
        )
    )
    target = sp.expand(curve.u_data - base_on_curve)

    # Need to invert arg_on_curve(s) = t to build F(t).
    t = sp.Symbol("t_fit", real=True)
    try:
        inv = sp.solve(sp.Eq(t, arg_on_curve), s, dict=False)
    except Exception:
        inv = []
    inv = list(inv) if isinstance(inv, (list, tuple)) else [inv]
    inv = [
        sol for sol in inv if sol is not None and s not in sp.sympify(sol).free_symbols
    ]
    if not inv:
        return None
    inv_s = min(inv, key=_expr_complexity)
    fitted = sp.Lambda(t, sp.expand(target.subs(s, inv_s)))
    new_rhs = sp.expand(rhs.subs(atom, fitted(arg_expr)))
    return sp.Eq(uexpr, new_rhs)


def _curve_parameter_samples(curve: InitialCurve2D, count=3):
    s = curve.parameter
    samples = [sp.Integer(i) for i in range(count)]
    return [{s: v} for v in samples]


def fit_complete_integral_to_initial_curve(
    solution_eq, initial, dep_expr_or_func, indep_vars=None
):
    """
    Restricted fitting of an explicit complete-integral/general-solution family to
    a two-dimensional initial curve.

    Supported cases:
      - one arbitrary function of one argument,
      - one or two free scalar parameters with polynomial coefficient matching in
        the curve parameter,
      - simple point-sample fitting fallback.
    """
    curve = process_initial_curve_2d(initial, dep_expr_or_func, indep_vars)
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    x, y = vars_
    rhs = sp.expand(solution_eq.rhs)

    fitted = _fit_single_free_function_family_2d(
        solution_eq, curve, dep_expr_or_func, vars_
    )
    if fitted is not None:
        return fitted

    # Scalar-parameter fit.
    params = sorted(
        list((rhs.free_symbols - set(vars_)) - {curve.parameter}), key=lambda s: s.name
    )
    if len(params) == 0:
        test_rhs = sp.expand(rhs.subs({x: curve.x_curve, y: curve.y_curve}))
        if sp.simplify(test_rhs - curve.u_data) == 0:
            return solution_eq
        return None

    expr_curve = sp.expand(
        rhs.subs({x: curve.x_curve, y: curve.y_curve}) - curve.u_data
    )

    # Polynomial coefficient matching in the curve parameter when possible.
    try:
        poly = sp.Poly(expr_curve, curve.parameter)
        coeffs = [sp.expand(c) for c in poly.all_coeffs()]
        if 1 <= len(params) <= 2:
            sol = sp.solve([sp.Eq(c, 0) for c in coeffs], params, dict=True)
            if sol:
                cand = sp.Eq(uexpr, sp.expand(rhs.subs(sol[0])))
                return cand
    except Exception:
        pass

    # Sample-point fallback for 1-2 parameters.
    if 1 <= len(params) <= 2:
        equations = []
        for sub in _curve_parameter_samples(curve, count=max(3, len(params))):
            lhsv = sp.expand(rhs.subs({x: curve.x_curve, y: curve.y_curve}).subs(sub))
            rhsv = sp.expand(curve.u_data.subs(sub))
            equations.append(sp.Eq(lhsv, rhsv))
        try:
            sol = sp.solve(equations, params, dict=True)
        except Exception:
            sol = []
        if sol:
            return sp.Eq(uexpr, sp.expand(rhs.subs(sol[0])))

    return None


def _verify_initial_curve_solution(
    solution_eq, curve: InitialCurve2D, dep_expr_or_func, indep_vars=None
):
    _, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    x, y = vars_
    rhs_on_curve = sp.expand(solution_eq.rhs.subs({x: curve.x_curve, y: curve.y_curve}))
    residual = sp.expand(rhs_on_curve - curve.u_data)
    try:
        ok = sp.simplify(residual) == 0
    except Exception:
        ok = False
    return ok, residual


def solve_first_order_cauchy_problem_2d(
    eq_or_expr,
    initial,
    dep_expr_or_func,
    indep_vars=None,
    *,
    assumptions=True,
    **kwargs,
):
    """
    Restricted first-order Cauchy-problem solver in two variables.

    Strategy:
      1. normalize the initial curve,
      2. try a complete-integral / general-family solver,
      3. fit the family to the initial curve,
      4. verify PDE and initial data.
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if len(vars_) != 2:
        raise ValueError(
            "Restricted first-order Cauchy solver is only implemented for two variables."
        )
    curve = process_initial_curve_2d(initial, uexpr, vars_)

    families = []
    details = {"attempts": []}

    # Try the complete-integral layer first.
    try:
        ci = solve_complete_integral_pde(
            eq_or_expr, uexpr, vars_, assumptions=assumptions, **kwargs
        )
        families.extend(ci.solutions)
        details["attempts"].append(("complete_integral", ci.method, len(ci.solutions)))
    except Exception as exc:
        details["attempts"].append(
            ("complete_integral_failed", type(exc).__name__, str(exc))
        )

    # Fall back to linear/quasilinear family solvers that may return arbitrary functions.
    from .classical_methods import PDEIVPResult, solve_first_order_linear_pde_pdsolve
    from .conservation_laws import solve_burgers_family
    from .classical_methods import solve_quasilinear_pde_characteristics_implicit

    for name, fn in [
        ("first_order_linear_pdsolve", solve_first_order_linear_pde_pdsolve),
        ("quasilinear_implicit", solve_quasilinear_pde_characteristics_implicit),
        ("burgers_family", solve_burgers_family),
    ]:
        try:
            out = fn(eq_or_expr, uexpr, vars_, **kwargs)
            sol = None
            if isinstance(out, PDEIVPResult):
                sol = out.solution if isinstance(out.solution, sp.Equality) else None
            elif isinstance(out, sp.Equality):
                sol = out
            if isinstance(sol, sp.Equality):
                families.append(sol)
                details["attempts"].append((name, "ok", 1))
                break
        except Exception as exc:
            details["attempts"].append((name + "_failed", type(exc).__name__, str(exc)))

    # Fit families to the initial curve.
    fitted = []
    verifs = []
    for fam in families:
        try:
            cand = fit_complete_integral_to_initial_curve(fam, initial, uexpr, vars_)
        except Exception:
            cand = None
        if cand is None:
            continue
        pde_ok, pde_res = _verify_first_order_solution(eq_or_expr, cand, uexpr, vars_)
        ic_ok, ic_res = _verify_initial_curve_solution(cand, curve, uexpr, vars_)
        if pde_ok and ic_ok:
            fitted.append(cand)
            verifs.append(
                {
                    "pde_verified": True,
                    "pde_residual": pde_res,
                    "initial_verified": True,
                    "initial_residual": ic_res,
                }
            )

    if not fitted:
        raise NotImplementedError(
            "Restricted first-order Cauchy solver could not fit a verified family to the initial curve."
        )

    return FirstOrderCauchyProblemResult(
        method="first_order_cauchy_curve_fit",
        solution=fitted[0],
        initial_curve=curve,
        details=details,
        verification=verifs[0],
    )
