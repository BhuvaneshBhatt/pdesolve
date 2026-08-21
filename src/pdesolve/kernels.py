from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from .domains import DomainGeometry, infer_domain_geometry
from .results import FundamentalSolutionResult, GreenFunctionResult
from .green_subsystem import (
    execute_advanced_green_plan,
    recognize_advanced_kernel_problem,
    AdvancedGreenPlan,
)


@dataclass(frozen=True)
class KernelMethodPlan:
    method: str
    operator_family: str
    geometry_kind: str
    boundary_family: str
    source_point: tuple[sp.Expr, ...]
    metadata: dict[str, Any]


def _as_eq(eq_or_expr):
    if isinstance(eq_or_expr, sp.Equality):
        return eq_or_expr
    return sp.Eq(sp.sympify(eq_or_expr), 0)


def _split_operator_and_source(eq: sp.Equality, dep_expr):
    zero = sp.expand(eq.lhs - eq.rhs)
    if zero.has(sp.DiracDelta):
        terms = list(sp.Add.make_args(zero))
        source_terms = [term for term in terms if term.has(sp.DiracDelta)]
        op_terms = [term for term in terms if not term.has(sp.DiracDelta)]
        op = sp.expand(sp.Add(*op_terms))
        source = sp.expand(-sp.Add(*source_terms))
        return op, source, 1
    return sp.expand(zero), None, 1


def _dirac_source_locations(source, vars_):
    if source is None:
        return None
    expr = sp.expand(source)
    deltas = list(expr.atoms(sp.DiracDelta))
    locs = []
    for v in vars_:
        found = None
        for d in deltas:
            arg = sp.expand(d.args[0])
            try:
                coeff = sp.diff(arg, v)
            except Exception:
                coeff = None
            if (
                coeff is not None
                and coeff != 0
                and all(not coeff.has(w) for w in vars_)
            ):
                shifted = sp.simplify(-arg.subs(v, 0) / coeff)
                found = shifted
                break
        if found is None:
            found = sp.Symbol(f"{v.name}_0", real=True)
        locs.append(found)
    return tuple(locs)


def _coeff(expr, term):
    try:
        return sp.simplify(sp.expand(expr).coeff(term))
    except Exception:
        return sp.S.Zero


def _match_heat_1d(op, dep, vars_):
    if len(vars_) != 2:
        return None
    x, t = vars_
    ut = sp.diff(dep, t)
    uxx = sp.diff(dep, x, 2)
    a_t = _coeff(op, ut)
    a_xx = _coeff(op, uxx)
    rem = sp.simplify(sp.expand(op - a_t * ut - a_xx * uxx))
    if a_t == 0 or a_xx == 0 or rem != 0:
        return None
    diffusivity = sp.simplify(-a_xx / a_t)
    if any(sp.sympify(diffusivity).has(v) for v in vars_):
        return None
    return {"family": "heat_1d", "diffusivity": diffusivity, "x": x, "t": t}


def _match_wave_1d(op, dep, vars_):
    if len(vars_) != 2:
        return None
    x, t = vars_
    utt = sp.diff(dep, t, 2)
    uxx = sp.diff(dep, x, 2)
    a_tt = _coeff(op, utt)
    a_xx = _coeff(op, uxx)
    rem = sp.simplify(sp.expand(op - a_tt * utt - a_xx * uxx))
    if a_tt == 0 or a_xx == 0 or rem != 0:
        return None
    c2 = sp.simplify(-a_xx / a_tt)
    if any(sp.sympify(c2).has(v) for v in vars_):
        return None
    return {"family": "wave_1d", "wave_speed": sp.sqrt(c2), "x": x, "t": t}


def _match_laplace_2d(op, dep, vars_):
    if len(vars_) != 2:
        return None
    x, y = vars_
    uxx = sp.diff(dep, x, 2)
    uyy = sp.diff(dep, y, 2)
    a_xx = _coeff(op, uxx)
    a_yy = _coeff(op, uyy)
    rem = sp.simplify(sp.expand(op - a_xx * uxx - a_yy * uyy))
    if a_xx == 0 or a_yy == 0 or rem != 0 or sp.simplify(a_xx - a_yy) != 0:
        return None
    if any(sp.sympify(a_xx).has(v) for v in vars_):
        return None
    return {"family": "laplace_2d", "scale": sp.simplify(a_xx), "x": x, "y": y}


def recognize_kernel_problem(eq_or_expr, dep_expr, vars_):
    eq = _as_eq(eq_or_expr)
    op, source, sign = _split_operator_and_source(eq, dep_expr)
    source_point = (
        _dirac_source_locations(source, vars_)
        if source is not None
        else tuple(sp.Symbol(f"{v.name}_0", real=True) for v in vars_)
    )
    heat = _match_heat_1d(op, dep_expr, tuple(vars_))
    wave = _match_wave_1d(op, dep_expr, tuple(vars_))
    lap = _match_laplace_2d(op, dep_expr, tuple(vars_))
    m = None
    if (
        len(vars_) == 2
        and lap is not None
        and (str(vars_[1]) != "t" or (heat is None and source is None))
    ):
        m = lap
    else:
        m = heat or wave or lap
    if m is not None:
        m["source_point"] = source_point
        m["has_source"] = source is not None
        m["source_sign"] = sign
        return m
    return None


def _coerce_geometry(
    vars_, *, geometry=None, bcs=None, condition_model=None, operator_family=None
):
    if isinstance(geometry, DomainGeometry):
        return geometry
    if isinstance(geometry, dict):
        return DomainGeometry(
            geometry.get("kind", "unspecified"),
            tuple(vars_),
            dict(geometry.get("extents", {}) or {}),
            dict(geometry.get("metadata", {}) or {}),
        )
    if isinstance(geometry, str):
        return DomainGeometry(geometry, tuple(vars_))
    geom = infer_domain_geometry(
        indep_vars=tuple(vars_), bcs=bcs, condition_model=condition_model
    )
    if (
        operator_family in {"heat_1d", "wave_1d"}
        and geom.kind == "unspecified_spacetime"
    ):
        return DomainGeometry("full_line", (vars_[0],), {"x": (-sp.oo, sp.oo)})
    if operator_family == "laplace_2d" and geom.kind == "unspecified_spacetime":
        return DomainGeometry("full_plane", tuple(vars_))
    return geom


def _boundary_family(bcs, condition_model, geom, vars_):
    text = repr(bcs)
    if "Dirichlet" in text or "dirichlet" in text:
        return "dirichlet"
    if "Neumann" in text or "neumann" in text:
        return "neumann"
    if condition_model is not None:
        from .conditions import summarize_condition_model

        kinds = set(
            summarize_condition_model(condition_model).get("boundary_kinds", ())
        )
        if "dirichlet" in kinds:
            return "dirichlet"
        if "neumann" in kinds:
            return "neumann"
    if geom.kind == "interval":
        return "dirichlet"
    return "free"


def build_kernel_method_plan(problem, *, geometry=None):
    details = getattr(problem.canonical_representation, "details", {}) or {}
    condition_model = problem.details.get("condition_model") or details.get(
        "condition_model"
    )
    preferred_family = getattr(
        getattr(problem, "profile", None), "canonical_family", None
    )
    domain_geometry = problem.details.get("domain_geometry") or details.get(
        "domain_geometry"
    )
    recog = recognize_kernel_problem(
        problem.equation, problem.dep_function, problem.indep_vars
    )
    if preferred_family == "laplace_like":
        op, source, sign = _split_operator_and_source(
            _as_eq(problem.equation), problem.dep_function
        )
        alt = _match_laplace_2d(op, problem.dep_function, tuple(problem.indep_vars))
        if alt is not None:
            alt["source_point"] = (
                _dirac_source_locations(source, problem.indep_vars)
                if source is not None
                else tuple(
                    sp.Symbol(f"{v.name}_0", real=True) for v in problem.indep_vars
                )
            )
            alt["has_source"] = source is not None
            alt["source_sign"] = sign
            recog = alt
    elif preferred_family == "wave_like":
        op, source, sign = _split_operator_and_source(
            _as_eq(problem.equation), problem.dep_function
        )
        alt = _match_wave_1d(op, problem.dep_function, tuple(problem.indep_vars))
        if alt is not None:
            alt["source_point"] = (
                _dirac_source_locations(source, problem.indep_vars)
                if source is not None
                else tuple(
                    sp.Symbol(f"{v.name}_0", real=True) for v in problem.indep_vars
                )
            )
            alt["has_source"] = source is not None
            alt["source_sign"] = sign
            recog = alt
    elif preferred_family == "heat_like":
        op, source, sign = _split_operator_and_source(
            _as_eq(problem.equation), problem.dep_function
        )
        alt = _match_heat_1d(op, problem.dep_function, tuple(problem.indep_vars))
        if alt is not None:
            alt["source_point"] = (
                _dirac_source_locations(source, problem.indep_vars)
                if source is not None
                else tuple(
                    sp.Symbol(f"{v.name}_0", real=True) for v in problem.indep_vars
                )
            )
            alt["has_source"] = source is not None
            alt["source_sign"] = sign
            recog = alt
    if isinstance(geometry, str) and geometry in {"half_plane", "full_plane"}:
        op, source, sign = _split_operator_and_source(
            _as_eq(problem.equation), problem.dep_function
        )
        alt = _match_laplace_2d(op, problem.dep_function, tuple(problem.indep_vars))
        if alt is not None:
            alt["source_point"] = (
                _dirac_source_locations(source, problem.indep_vars)
                if source is not None
                else tuple(
                    sp.Symbol(f"{v.name}_0", real=True) for v in problem.indep_vars
                )
            )
            alt["has_source"] = source is not None
            alt["source_sign"] = sign
            recog = alt
    geom_probe = _coerce_geometry(
        problem.indep_vars,
        geometry=geometry or domain_geometry,
        bcs=problem.bcs,
        condition_model=condition_model,
        operator_family=(recog or {}).get("family")
        if isinstance(recog, dict)
        else None,
    )
    prefer_advanced_kinds = {"strip", "semi_infinite_strip", "quadrant", "half_space"}
    use_advanced = (
        recog is None or getattr(geom_probe, "kind", None) in prefer_advanced_kinds
    )
    if use_advanced:
        adv = recognize_advanced_kernel_problem(
            problem.equation, problem.dep_function, problem.indep_vars
        )
        if adv is not None:
            geom = _coerce_geometry(
                problem.indep_vars,
                geometry=geometry or domain_geometry,
                bcs=problem.bcs,
                condition_model=condition_model,
                operator_family=adv["family"],
            )
            boundary = _boundary_family(
                problem.bcs, condition_model, geom, problem.indep_vars
            )
            free_kinds = {"free", "full_line", "full_plane", "full_space"}
            method = (
                "kernel_fundamental_solution"
                if boundary == "free" and geom.kind in free_kinds
                else "kernel_green_function"
            )
            return KernelMethodPlan(
                method=method,
                operator_family=adv["family"],
                geometry_kind=geom.kind,
                boundary_family=boundary,
                source_point=tuple(adv["source_point"]),
                metadata={
                    "recognition": adv,
                    "geometry": geom,
                    "condition_model": condition_model,
                    "subsystem": "advanced_green",
                    "advanced_plan": AdvancedGreenPlan(
                        method="advanced_fundamental_solution"
                        if method == "kernel_fundamental_solution"
                        else "advanced_green_function",
                        operator_family=adv["family"],
                        geometry_kind=geom.kind,
                        boundary_family=boundary,
                        source_point=tuple(adv["source_point"]),
                        metadata={
                            "recognition": adv,
                            "geometry": geom,
                            "condition_model": condition_model,
                        },
                    ),
                },
            )
        if recog is None:
            return None
    geom = _coerce_geometry(
        problem.indep_vars,
        geometry=geometry or domain_geometry,
        bcs=problem.bcs,
        condition_model=condition_model,
        operator_family=recog["family"],
    )
    boundary = _boundary_family(problem.bcs, condition_model, geom, problem.indep_vars)
    method = (
        "kernel_fundamental_solution"
        if boundary == "free" and geom.kind in {"full_line", "full_plane"}
        else "kernel_green_function"
    )
    return KernelMethodPlan(
        method=method,
        operator_family=recog["family"],
        geometry_kind=geom.kind,
        boundary_family=boundary,
        source_point=tuple(recog["source_point"]),
        metadata={
            "recognition": recog,
            "geometry": geom,
            "condition_model": condition_model,
        },
    )


def _heat_kernel_line(a, x, t, xi, tau):
    return (
        sp.Heaviside(t - tau)
        * sp.exp(-((x - xi) ** 2) / (4 * a * (t - tau)))
        / sp.sqrt(4 * sp.pi * a * (t - tau))
    )


def _heat_kernel_half_line(a, x, t, xi, tau, *, boundary="dirichlet"):
    base = _heat_kernel_line(a, x, t, xi, tau)
    image = _heat_kernel_line(a, x, t, -xi, tau)
    return sp.simplify(base - image if boundary == "dirichlet" else base + image)


def _heat_kernel_interval(a, x, t, xi, tau, L, *, boundary="dirichlet"):
    n = sp.Symbol("n", integer=True, positive=True)
    if boundary == "neumann":
        return sp.Heaviside(t - tau) * (
            sp.Rational(1, 1) / L
            + (2 / L)
            * sp.Sum(
                sp.cos(n * sp.pi * x / L)
                * sp.cos(n * sp.pi * xi / L)
                * sp.exp(-a * n**2 * sp.pi**2 * (t - tau) / L**2),
                (n, 1, sp.oo),
            )
        )
    return (
        sp.Heaviside(t - tau)
        * (2 / L)
        * sp.Sum(
            sp.sin(n * sp.pi * x / L)
            * sp.sin(n * sp.pi * xi / L)
            * sp.exp(-a * n**2 * sp.pi**2 * (t - tau) / L**2),
            (n, 1, sp.oo),
        )
    )


def _wave_kernel_line(c, x, t, xi, tau):
    return sp.Heaviside((t - tau) - sp.Abs(x - xi) / c) / (2 * c)


def _wave_kernel_half_line(c, x, t, xi, tau, *, boundary="dirichlet"):
    base = _wave_kernel_line(c, x, t, xi, tau)
    image = _wave_kernel_line(c, x, t, -xi, tau)
    return sp.simplify(base - image if boundary == "dirichlet" else base + image)


def _wave_kernel_interval(c, x, t, xi, tau, L, *, boundary="dirichlet"):
    n = sp.Symbol("n", integer=True, positive=True)
    if boundary == "neumann":
        return sp.Heaviside(t - tau) * (
            (t - tau) / L
            + (2 / (sp.pi * c))
            * sp.Sum(
                sp.cos(n * sp.pi * x / L)
                * sp.cos(n * sp.pi * xi / L)
                * sp.sin(c * n * sp.pi * (t - tau) / L)
                / n,
                (n, 1, sp.oo),
            )
        )
    return (
        sp.Heaviside(t - tau)
        * (2 / (sp.pi * c))
        * sp.Sum(
            sp.sin(n * sp.pi * x / L)
            * sp.sin(n * sp.pi * xi / L)
            * sp.sin(c * n * sp.pi * (t - tau) / L)
            / n,
            (n, 1, sp.oo),
        )
    )


def _laplace_green_free(scale, x, y, xi, eta):
    return -sp.log((x - xi) ** 2 + (y - eta) ** 2) / (4 * sp.pi * scale)


def _laplace_green_half_plane(scale, x, y, xi, eta, *, boundary="dirichlet"):
    if boundary == "neumann":
        return -(
            sp.log((x - xi) ** 2 + (y - eta) ** 2)
            + sp.log((x - xi) ** 2 + (y + eta) ** 2)
        ) / (4 * sp.pi * scale)
    return sp.log(
        ((x - xi) ** 2 + (y + eta) ** 2) / ((x - xi) ** 2 + (y - eta) ** 2)
    ) / (4 * sp.pi * scale)


def solve_fundamental_solution(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    assumptions=True,
    source_point=None,
):
    from .problem import build_pde_problem

    problem = build_pde_problem(
        eq_or_expr, dep_expr_or_func, indep_vars, assumptions=assumptions
    )
    plan = build_kernel_method_plan(
        problem,
        geometry="full_plane"
        if len(problem.indep_vars) == 2
        and problem.profile
        and problem.profile.canonical_family == "laplace_like"
        else None,
    )
    if plan is not None:
        return execute_kernel_plan(
            problem,
            plan=KernelMethodPlan(
                method="kernel_fundamental_solution",
                operator_family=plan.operator_family,
                geometry_kind="full_plane"
                if plan.operator_family == "laplace_2d"
                else "full_line",
                boundary_family="free",
                source_point=source_point or plan.source_point,
                metadata=plan.metadata,
            ),
        )
    return execute_advanced_green_plan(
        eq_or_expr,
        dep_expr_or_func,
        problem.indep_vars,
        assumptions=assumptions,
        geometry="full_space",
    )


def solve_green_function(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    bcs=None,
    assumptions=True,
    geometry=None,
    source_point=None,
):
    from .problem import build_pde_problem

    problem = build_pde_problem(
        eq_or_expr, dep_expr_or_func, indep_vars, bcs=bcs, assumptions=assumptions
    )
    plan = build_kernel_method_plan(problem, geometry=geometry)
    if plan is not None:
        if source_point is not None:
            plan = KernelMethodPlan(
                method=plan.method,
                operator_family=plan.operator_family,
                geometry_kind=plan.geometry_kind,
                boundary_family=plan.boundary_family,
                source_point=tuple(source_point),
                metadata=plan.metadata,
            )
        return execute_kernel_plan(problem, plan=plan)
    return execute_advanced_green_plan(
        eq_or_expr,
        dep_expr_or_func,
        problem.indep_vars,
        bcs=bcs,
        assumptions=assumptions,
        geometry=geometry,
    )


def execute_kernel_plan(problem, *, plan: KernelMethodPlan | None = None):
    plan = plan or build_kernel_method_plan(problem)
    if plan is None:
        return execute_advanced_green_plan(
            problem.equation,
            problem.dep_function,
            problem.indep_vars,
            bcs=problem.bcs,
            assumptions=problem.assumptions,
            geometry=problem.details.get("domain_geometry"),
        )
    if (plan.metadata or {}).get("subsystem") == "advanced_green" or (
        plan.metadata or {}
    ).get("advanced_plan") is not None:
        return execute_advanced_green_plan(
            problem.equation,
            problem.dep_function,
            problem.indep_vars,
            bcs=problem.bcs,
            assumptions=problem.assumptions,
            geometry=(plan.metadata or {}).get("geometry")
            or problem.details.get("domain_geometry"),
        )
    recog = dict(plan.metadata.get("recognition", {}) or {})
    family = plan.operator_family
    geom_kind = plan.geometry_kind
    boundary = plan.boundary_family
    if family == "heat_1d":
        a = recog["diffusivity"]
        x = recog["x"]
        t = recog["t"]
        xi, tau = plan.source_point[:2]
        if geom_kind == "full_line":
            kernel = _heat_kernel_line(a, x, t, xi, tau)
        elif geom_kind == "half_line":
            kernel = _heat_kernel_half_line(a, x, t, xi, tau, boundary=boundary)
        elif geom_kind == "interval":
            geom = plan.metadata.get("geometry")
            L = None
            if geom is not None:
                ext = geom.extents.get("x") if hasattr(geom, "extents") else None
                if isinstance(ext, tuple) and len(ext) == 2:
                    L = sp.simplify(ext[1] - ext[0])
            L = L or sp.Symbol("L", positive=True)
            kernel = _heat_kernel_interval(
                a,
                x,
                t,
                xi,
                tau,
                L,
                boundary="neumann" if boundary == "neumann" else "dirichlet",
            )
        else:
            raise NotImplementedError(f"Unsupported heat geometry: {geom_kind}")
    elif family == "wave_1d":
        c = recog["wave_speed"]
        x = recog["x"]
        t = recog["t"]
        xi, tau = plan.source_point[:2]
        if geom_kind == "full_line":
            kernel = _wave_kernel_line(c, x, t, xi, tau)
        elif geom_kind == "half_line":
            kernel = _wave_kernel_half_line(c, x, t, xi, tau, boundary=boundary)
        elif geom_kind == "interval":
            geom = plan.metadata.get("geometry")
            L = None
            if geom is not None:
                ext = geom.extents.get("x") if hasattr(geom, "extents") else None
                if isinstance(ext, tuple) and len(ext) == 2:
                    L = sp.simplify(ext[1] - ext[0])
            L = L or sp.Symbol("L", positive=True)
            kernel = _wave_kernel_interval(
                c,
                x,
                t,
                xi,
                tau,
                L,
                boundary="neumann" if boundary == "neumann" else "dirichlet",
            )
        else:
            raise NotImplementedError(f"Unsupported wave geometry: {geom_kind}")
    elif family == "laplace_2d":
        scale = recog["scale"]
        x = recog["x"]
        y = recog["y"]
        xi, eta = plan.source_point[:2]
        if geom_kind in {"full_plane", "full_space"}:
            kernel = _laplace_green_free(scale, x, y, xi, eta)
        elif geom_kind == "half_plane":
            kernel = _laplace_green_half_plane(scale, x, y, xi, eta, boundary=boundary)
        else:
            raise NotImplementedError(f"Unsupported Laplace geometry: {geom_kind}")
    else:
        raise NotImplementedError(f"Unsupported operator family: {family}")

    meta = {
        "operator_family": family,
        "geometry_kind": geom_kind,
        "boundary_family": boundary,
        "source_point": plan.source_point,
        "kernel_plan": plan,
    }
    if plan.method == "kernel_fundamental_solution":
        return FundamentalSolutionResult(
            method=plan.method,
            solution=kernel,
            classification=family,
            metadata=meta,
            operator_family=family,
            kernel=kernel,
            source_point=plan.source_point,
        )
    return GreenFunctionResult(
        method=plan.method,
        solution=kernel,
        classification=family,
        metadata=meta,
        operator_family=family,
        kernel=kernel,
        source_point=plan.source_point,
        boundary_type=boundary,
    )


__all__ = [
    "KernelMethodPlan",
    "build_kernel_method_plan",
    "solve_fundamental_solution",
    "solve_green_function",
    "execute_kernel_plan",
    "recognize_kernel_problem",
]
