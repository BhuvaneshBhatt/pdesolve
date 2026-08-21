from __future__ import annotations

import sympy as sp

from .conditions import (
    extract_equations_by_role,
    select_boundary_equations,
    summarize_condition_model,
)
from .dispatcher_support import as_verification_summary
from .errors import PDEInputError, PDEMethodNotApplicable
from .method_names import normalize_method_name
from .problem import PDEProblem, build_pde_problem, build_system_pde_problem
from .results import PDESolutionRecord, PDEVerificationSummary
from .separation_framework import execute_separation_plan
from .solve_pipeline import solve_scalar_problem, solve_system_problem
from .transform_framework import execute_transform_plan


def _classical():
    from . import classical_methods as classical_mod

    return classical_mod


def _derivative_order_in_expr(expr, dep_expr_or_func, vars_):
    try:
        zero = (expr.lhs - expr.rhs) if isinstance(expr, sp.Equality) else expr
    except Exception:
        zero = expr
    try:
        orders = []
        dep_func = dep_expr_or_func.func if hasattr(dep_expr_or_func, "func") else dep_expr_or_func
        for d in zero.atoms(sp.Derivative):
            try:
                if getattr(d.expr, "func", None) == dep_func:
                    orders.append(sum(k for _, k in d.variable_count))
            except Exception:
                pass
        if orders:
            return max(orders)
        if zero.has(dep_func):
            return 0
    except Exception:
        pass
    return -1


def _is_constant_wrt(expr, syms):
    try:
        return not any(sp.sympify(expr).has(s) for s in syms)
    except Exception:
        return False


def _bundle_conditions_to_ic_bc(conditions, dep_expr_or_func, vars_):
    if not conditions:
        return None, None
    dep_func = dep_expr_or_func.func if hasattr(dep_expr_or_func, "func") else dep_expr_or_func
    ics_eqs = []
    bc_eqs = []
    ics = {"equations": []}
    bcs = {"equations": []}
    v1 = vars_[0] if len(vars_) >= 1 else None
    v2 = vars_[1] if len(vars_) >= 2 else None
    bc_markers = []
    for eq in conditions:
        if not isinstance(eq, sp.Equality):
            continue
        dep_side = None
        other = None
        for s in (eq.lhs, eq.rhs):
            if isinstance(s, sp.Derivative):
                if getattr(s.expr, "func", None) == dep_func:
                    dep_side = s
                    other = eq.rhs if s == eq.lhs else eq.lhs
                    break
            elif getattr(s, "func", None) == dep_func:
                dep_side = s
                other = eq.rhs if s == eq.lhs else eq.lhs
                break
        if dep_side is None:
            bc_eqs.append(eq)
            bcs["equations"].append(eq)
            continue
        base = dep_side.expr if isinstance(dep_side, sp.Derivative) else dep_side
        args = getattr(base, "args", ())
        if len(vars_) == 2 and len(args) == 2:
            a1, a2 = args
            order_v1 = 0
            order_v2 = 0
            if isinstance(dep_side, sp.Derivative):
                for vv, kk in dep_side.variable_count:
                    if vv == v1:
                        order_v1 += kk
                    if vv == v2:
                        order_v2 += kk
            if a1 == v1 and _is_constant_wrt(a2, vars_):
                ics_eqs.append(eq)
                ics["equations"].append(eq)
                ics.setdefault("curve_value", sp.sympify(a2))
                if order_v2 > 0:
                    if order_v2 == 1:
                        ics["initial_velocity"] = other
                    else:
                        ics.setdefault("higher_initial_derivatives", []).append((order_v2, other))
                else:
                    if "initial_profile" not in ics:
                        ics["initial_profile"] = other
                        ics.setdefault("equation", eq)
                        ics.setdefault("initial_equation", eq)
                    else:
                        ics["initial_displacement"] = other
                continue
            if a2 == v2 and _is_constant_wrt(a1, vars_):
                bc_eqs.append(eq)
                bcs["equations"].append(eq)
                kind = "neumann" if order_v1 > 0 else "dirichlet"
                bc_markers.append((kind, sp.sympify(a1), other))
                continue
        bc_eqs.append(eq)
        bcs["equations"].append(eq)
    if bc_markers and len(vars_) >= 2:
        kinds = {k for k, _, _ in bc_markers}
        locs = [loc for _, loc, _ in bc_markers]
        vals = [val for _, _, val in bc_markers]
        if len(bc_markers) == 1 and all(sp.simplify(v) == 0 for v in vals):
            if "dirichlet" in kinds:
                bcs.setdefault("type", "half_line_dirichlet")
                bcs.setdefault("value", vals[0])
            elif "neumann" in kinds:
                bcs.setdefault("type", "half_line_neumann")
                bcs.setdefault("value", vals[0])
        elif (
            len(bc_markers) >= 2 and all(sp.simplify(v) == 0 for v in vals) and len(set(locs)) >= 2
        ):
            try:
                L = max(set(locs), key=lambda z: sp.default_sort_key(z))
            except Exception:
                L = list(set(locs))[-1]
            if kinds == {"dirichlet"}:
                bcs.setdefault("type", "dirichlet_homogeneous_interval")
                bcs.setdefault("length", L)
            elif kinds == {"neumann"}:
                bcs.setdefault("type", "neumann_homogeneous_interval")
                bcs.setdefault("length", L)
    if ics == {"equations": []}:
        ics = None
    if bcs == {"equations": []}:
        bcs = None
    return ics, bcs


def _split_scalar_pde_bundle(eq_or_expr, dep_expr_or_func, indep_vars):
    if not isinstance(eq_or_expr, (list, tuple)):
        return eq_or_expr, None, None
    vars_ = (
        tuple(indep_vars)
        if indep_vars is not None
        else tuple(dep_expr_or_func.args if hasattr(dep_expr_or_func, "args") else ())
    )
    items = list(eq_or_expr)
    if not items:
        return eq_or_expr, None, None
    scored = [
        (_derivative_order_in_expr(item, dep_expr_or_func, vars_), i, item)
        for i, item in enumerate(items)
    ]
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    pde_eq = scored[0][2]
    rest = [item for item in items if item is not pde_eq]
    ics, bcs = _bundle_conditions_to_ic_bc(rest, dep_expr_or_func, vars_)
    return pde_eq, ics, bcs


def _split_system_bundle(eqns_or_bundle, dep_exprs_or_funcs, indep_vars):
    if not isinstance(eqns_or_bundle, (list, tuple)):
        return eqns_or_bundle, []
    funcs = list(dep_exprs_or_funcs)
    vars_ = tuple(indep_vars)
    items = list(eqns_or_bundle)
    if not items:
        return items, []

    def score(item):
        best = -1
        for f in funcs:
            best = max(best, _derivative_order_in_expr(item, f, vars_))
        return best

    scored = [(score(item), i, item) for i, item in enumerate(items)]
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    n_eq = min(len(funcs), len(items))
    pde_indices = {t[1] for t in scored[:n_eq]}
    pde_eqs = [items[i] for i in range(len(items)) if i in pde_indices]
    rest = [items[i] for i in range(len(items)) if i not in pde_indices]
    return pde_eqs, rest


def _problem_separation_plan(problem):
    if getattr(problem, "canonical_representation", None) is not None:
        details = getattr(problem.canonical_representation, "details", {}) or {}
        spn = details.get("separation_plan")
        if spn is not None:
            return spn
    return problem.details.get("separation_plan") if getattr(problem, "details", None) else None


def _problem_transform_plan(problem):
    if getattr(problem, "canonical_representation", None) is not None:
        details = getattr(problem.canonical_representation, "details", {}) or {}
        tp = details.get("transform_plan")
        if tp is not None:
            return tp
    return problem.details.get("transform_plan") if getattr(problem, "details", None) else None


def _problem_boundary_model(problem):
    if getattr(problem, "canonical_representation", None) is not None:
        details = getattr(problem.canonical_representation, "details", {}) or {}
        bm = details.get("boundary_model")
        if bm is not None:
            return bm
    return problem.details.get("boundary_model") if getattr(problem, "details", None) else None


def _structured_interval_length(problem):
    geom = _problem_domain_geometry(problem)
    if geom is None:
        return None
    for ext in getattr(geom, "extents", {}).values():
        if isinstance(ext, tuple) and len(ext) == 2:
            try:
                return sp.simplify(ext[1] - ext[0])
            except Exception:
                return ext[1]
    return None


def _structured_rectangle_lengths(problem):
    geom = _problem_domain_geometry(problem)
    if geom is None:
        return None, None
    xext = geom.extents.get("x")
    yext = geom.extents.get("y")
    xl = sp.simplify(xext[1] - xext[0]) if isinstance(xext, tuple) and len(xext) == 2 else None
    yl = sp.simplify(yext[1] - yext[0]) if isinstance(yext, tuple) and len(yext) == 2 else None
    return xl, yl


def _structured_boundary_top(problem, yvar):
    cm = _problem_condition_model(problem)
    geom = _problem_domain_geometry(problem)
    if cm is None or geom is None:
        return None
    yext = geom.extents.get("y")
    if not (isinstance(yext, tuple) and len(yext) == 2):
        return None
    eqs = select_boundary_equations(cm, variable=yvar, location=yext[1], kind="dirichlet")
    if eqs:
        return eqs[0].rhs
    return None


def _solve_from_separation_plan(problem, classical_mod, **kwargs):
    return execute_separation_plan(problem, classical_mod, **kwargs)


def _solve_from_transform_plan(problem, **kwargs):
    classical_mod = _classical()
    return execute_transform_plan(problem, classical_mod=classical_mod, **kwargs)


def _classify_only_result(problem: PDEProblem):
    profile = problem.profile
    classification = profile.second_order_class
    return PDESolutionRecord(
        method="classification_only",
        solution=problem.equation,
        steps=("classification_only",),
        verification=PDEVerificationSummary(
            None, "unverified", mode="structural", message="classification-only result"
        ).as_dict(),
        assumptions=problem.assumptions,
        canonical_equation=problem.equation,
        metadata={"classification": classification, "profile": profile},
    )


def pdesolve(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    method="auto",
    assumptions=True,
    **kwargs,
):
    classical_mod = _classical()
    method = normalize_method_name(method)
    normalize_result = kwargs.pop("normalize_result", True)
    max_ops = kwargs.pop("normalization_max_ops", 80)
    domain = kwargs.pop("domain", None)

    if not isinstance(dep_expr_or_func, (list, tuple)) and isinstance(eq_or_expr, (list, tuple)):
        eq_or_expr, bundled_ics, bundled_bcs = _split_scalar_pde_bundle(
            eq_or_expr,
            dep_expr_or_func,
            indep_vars,
        )
        if ics is None:
            ics = bundled_ics
        if bcs is None:
            bcs = bundled_bcs

    if isinstance(dep_expr_or_func, (list, tuple)):
        variables = tuple(indep_vars or ())
        if len(variables) != 2:
            raise PDEInputError("Hyperbolic system solving expects two independent variables.")
        if method not in {"auto", "hyperbolic_system"}:
            raise PDEMethodNotApplicable(
                "System problems support auto or hyperbolic_system methods."
            )
        equations, bundled_ics = _split_system_bundle(eq_or_expr, dep_expr_or_func, variables)
        if isinstance(ics, dict):
            initial_eqs = list(ics.get("equations", ()) or ())
        elif isinstance(ics, (list, tuple)):
            initial_eqs = list(ics)
        else:
            initial_eqs = list(bundled_ics or ())
        problem = build_system_pde_problem(
            equations,
            dep_expr_or_func,
            variables,
            ics=initial_eqs,
            assumptions=assumptions,
        )
        return solve_system_problem(
            problem,
            dependent=tuple(dep_expr_or_func),
            variables=variables,
            ics=initial_eqs,
            assumptions=assumptions,
            normalize_result=normalize_result,
            max_ops=max_ops,
            solver_kwargs=kwargs,
        )

    dependent, variables = classical_mod._dep_and_vars(dep_expr_or_func, indep_vars)
    problem = build_pde_problem(
        eq_or_expr,
        dependent,
        variables,
        ics=ics,
        bcs=bcs,
        domain=domain,
        assumptions=assumptions,
    )
    return solve_scalar_problem(
        problem,
        dependent=dependent,
        variables=variables,
        ics=ics,
        bcs=bcs,
        method=method,
        assumptions=assumptions,
        normalize_result=normalize_result,
        max_ops=max_ops,
        solver_kwargs=kwargs,
    )


def extract_solution_trace(record):
    metadata = getattr(record, "metadata", None)
    if metadata is not None:
        return metadata.get("trace")
    return None


def summarize_solution_record(record):
    trace = extract_solution_trace(record)
    verification = as_verification_summary(getattr(record, "verification", {}))
    return {
        "method": getattr(record, "method", None),
        "verified": verification.verified,
        "status": verification.status,
        "attempted_methods": getattr(trace, "attempted_methods", ()),
        "selected_method": getattr(trace, "selected_method", None),
    }


def _problem_condition_model(problem):
    if getattr(problem, "canonical_representation", None) is not None:
        details = getattr(problem.canonical_representation, "details", {}) or {}
        cm = details.get("condition_model")
        if cm is not None:
            return cm
    return problem.details.get("condition_model") if getattr(problem, "details", None) else None


def _problem_domain_geometry(problem):
    if getattr(problem, "canonical_representation", None) is not None:
        details = getattr(problem.canonical_representation, "details", {}) or {}
        dg = details.get("domain_geometry")
        if dg is not None:
            return dg
    return problem.details.get("domain_geometry") if getattr(problem, "details", None) else None


def _extract_condition_payloads(problem):
    cm = _problem_condition_model(problem)
    summary = summarize_condition_model(cm) if cm is not None else {}
    ics_payload = dict(problem.ics) if isinstance(problem.ics, dict) else {}
    bcs_payload = dict(problem.bcs) if isinstance(problem.bcs, dict) else {}
    if cm is not None:
        init_eqs = extract_equations_by_role(cm, "initial")
        bc_eqs = extract_equations_by_role(cm, "boundary")
        if "equations" not in ics_payload and init_eqs:
            ics_payload = {**ics_payload, "equations": init_eqs}
        if "equations" not in bcs_payload and bc_eqs:
            bcs_payload = {**bcs_payload, "equations": bc_eqs}
        if init_eqs and "equation" not in ics_payload:
            ics_payload = {**ics_payload, "equation": init_eqs[0]}
            ics_payload.setdefault("initial_equation", init_eqs[0])
        ts = summary.get("time_slices", ())
        if ts and "curve_value" not in ics_payload:
            ics_payload["curve_value"] = ts[0]
        # Derive a simple initial profile / displacement / velocity payload from structured conditions.
        for cond, ck in zip(
            getattr(cm, "initial_conditions", ()), summary.get("initial_kinds", ()), strict=True
        ):
            rhs = cond.equation.rhs if isinstance(cond.equation, sp.Equality) else None
            if ck == "profile" and rhs is not None and "initial_profile" not in ics_payload:
                ics_payload["initial_profile"] = rhs
                ics_payload.setdefault("initial_displacement", rhs)
            elif ck == "velocity" and rhs is not None and "initial_velocity" not in ics_payload:
                ics_payload["initial_velocity"] = rhs
        # Attach geometric hints.
        geom = problem.details.get("domain_geometry")
        if geom is not None:
            if "type" not in bcs_payload:
                if geom.kind == "interval" and set(summary.get("boundary_kinds", ())) == {
                    "dirichlet"
                }:
                    bcs_payload["type"] = "dirichlet_homogeneous_interval"
                elif geom.kind == "interval" and set(summary.get("boundary_kinds", ())) == {
                    "neumann"
                }:
                    bcs_payload["type"] = "neumann_homogeneous_interval"
            for key, val in getattr(geom, "extents", {}).items():
                if key == "x" and isinstance(val, tuple) and len(val) == 2:
                    bcs_payload.setdefault("length", val[1] - val[0])
                else:
                    bcs_payload.setdefault(f"{key}_extent", val)
    return cm, summary, ics_payload, bcs_payload
