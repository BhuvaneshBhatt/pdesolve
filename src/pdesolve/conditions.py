from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class ConditionEquation:
    """Canonical representation of one initial/boundary condition.

    ``role`` identifies where the data live (initial, boundary, mixed, event),
    while ``kind`` records the mathematical condition type after parsing.
    Existing code that only used ``role``, ``variable`` and ``location``
    remains source-compatible.
    """

    equation: sp.Equality
    role: str  # initial | boundary | mixed | event | unknown
    variable: sp.Symbol | None = None
    location: sp.Expr | None = None
    derivative_multiindex: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str | None = None

    @property
    def surface(self) -> tuple[sp.Symbol | None, sp.Expr | None]:
        return self.variable, self.location

    @property
    def derivative_order(self) -> int:
        return sum(self.derivative_multiindex or ())


# Semantic subclasses are useful to downstream callers while preserving the
# ConditionEquation contract used by the pre-0.1 API.
class InitialCondition(ConditionEquation):
    pass


class DirichletCondition(ConditionEquation):
    pass


class NeumannCondition(ConditionEquation):
    pass


class RobinCondition(ConditionEquation):
    pass


class PeriodicCondition(ConditionEquation):
    pass


class InterfaceCondition(ConditionEquation):
    pass


class AsymptoticCondition(ConditionEquation):
    pass


@dataclass(frozen=True)
class ConditionModel:
    dependent_function: Any
    independent_variables: tuple[sp.Symbol, ...]
    initial_conditions: tuple[ConditionEquation, ...] = ()
    boundary_conditions: tuple[ConditionEquation, ...] = ()
    mixed_conditions: tuple[ConditionEquation, ...] = ()
    event_conditions: tuple[ConditionEquation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_conditions(self) -> tuple[ConditionEquation, ...]:
        return (
            self.initial_conditions
            + self.boundary_conditions
            + self.mixed_conditions
            + self.event_conditions
        )


def _as_eq(obj: Any) -> sp.Equality | None:
    return obj if isinstance(obj, sp.Equality) else None


def _dep_func(dep_expr):
    return getattr(dep_expr, "func", dep_expr)


def _lhs_target_expr(
    expr: sp.Expr, dep_func, indep_vars: tuple[sp.Symbol, ...]
) -> tuple[tuple[sp.Expr, ...] | None, tuple[int, ...]]:
    if (
        isinstance(expr, sp.Subs)
        and isinstance(expr.expr, sp.Derivative)
        and getattr(expr.expr.expr, "func", None) == dep_func
    ):
        args = list(expr.expr.expr.args)
        for old, new in zip(expr.variables, expr.point):
            args = [new if a == old else a for a in args]
        multi = []
        for v in expr.expr.expr.args:
            multi.append(
                next((count for vv, count in expr.expr.variable_count if vv == v), 0)
            )
        return tuple(args), tuple(multi)
    if isinstance(expr, sp.Derivative) and getattr(expr.expr, "func", None) == dep_func:
        args = tuple(expr.expr.args)
        multi = tuple(
            next((count for vv, count in expr.variable_count if vv == v), 0)
            for v in args
        )
        return args, multi
    if getattr(expr, "func", None) == dep_func:
        return tuple(expr.args), tuple(0 for _ in expr.args)
    return None, ()


def _lhs_target(
    eq: sp.Equality, dep_func
) -> tuple[tuple[sp.Expr, ...] | None, tuple[int, ...]]:
    return _lhs_target_expr(eq.lhs, dep_func, tuple(getattr(eq.lhs, "args", ())))


def _fixed_slice(args, indep_vars):
    fixed = []
    for v, a in zip(indep_vars, args):
        if a == v:
            continue
        if not sp.sympify(a).has(*indep_vars):
            fixed.append((v, sp.simplify(a)))
        else:
            return None
    return tuple(fixed)


def _linear_boundary_terms(lhs: sp.Expr, dep_func, indep_vars: tuple[sp.Symbol, ...]):
    """Recognize alpha*u + beta*d(u)/dn on one coordinate boundary."""
    terms = sp.Add.make_args(sp.expand(lhs))
    parsed = []
    for term in terms:
        matched = None
        # A substituted normal derivative contains both Subs and the underlying
        # unfixed Derivative atom; inspect Subs first so the boundary location
        # is not lost.
        for atom in sorted(term.atoms(sp.Subs), key=sp.default_sort_key):
            args, multi = _lhs_target_expr(atom, dep_func, indep_vars)
            if args is not None:
                coeff = sp.simplify(term / atom)
                matched = (args, multi, coeff)
                break
        if matched is None:
            for atom in sorted(term.atoms(sp.Derivative), key=sp.default_sort_key):
                args, multi = _lhs_target_expr(atom, dep_func, indep_vars)
                if args is not None:
                    coeff = sp.simplify(term / atom)
                    matched = (args, multi, coeff)
                    break
        if matched is None:
            for atom in term.atoms(sp.Function):
                if getattr(atom, "func", None) == dep_func:
                    args, multi = _lhs_target_expr(atom, dep_func, indep_vars)
                    coeff = sp.simplify(term / atom)
                    matched = (args, multi, coeff)
                    break
        if matched is None:
            if sp.simplify(term) != 0:
                return None
        else:
            parsed.append(matched)
    if not parsed:
        return None
    fixed_sets = [_fixed_slice(args, indep_vars) for args, _, _ in parsed]
    if (
        any(fs is None or len(fs) != 1 for fs in fixed_sets)
        or len(set(fixed_sets)) != 1
    ):
        return None
    ((var, loc),) = fixed_sets[0]
    alpha = sp.Integer(0)
    beta = sp.Integer(0)
    derivative_multiindex = tuple(0 for _ in indep_vars)
    for _, multi, coeff in parsed:
        if sum(multi) == 0:
            alpha += coeff
        elif sum(multi) == 1 and multi[indep_vars.index(var)] == 1:
            beta += coeff
            derivative_multiindex = multi
        else:
            return None
    if sp.simplify(beta) == 0:
        return None
    return var, loc, sp.simplify(alpha), sp.simplify(beta), derivative_multiindex


def _condition_from_eq(
    eq: sp.Equality, dep_func, indep_vars: tuple[sp.Symbol, ...]
) -> ConditionEquation | None:
    # Periodic data: u(a, ...) = u(b, ...) (or matching same derivatives).
    largs, lmulti = _lhs_target_expr(eq.lhs, dep_func, indep_vars)
    rargs, rmulti = _lhs_target_expr(eq.rhs, dep_func, indep_vars)
    if largs is not None and rargs is not None and lmulti == rmulti:
        lf, rf = _fixed_slice(largs, indep_vars), _fixed_slice(rargs, indep_vars)
        if (
            lf
            and rf
            and len(lf) == len(rf) == 1
            and lf[0][0] == rf[0][0]
            and sp.simplify(lf[0][1] - rf[0][1]) != 0
        ):
            var, loc = lf[0]
            return PeriodicCondition(
                eq,
                "boundary",
                variable=var,
                location=loc,
                derivative_multiindex=lmulti,
                metadata={"paired_location": rf[0][1]},
                kind="periodic",
            )

    args, multi = _lhs_target_expr(eq.lhs, dep_func, indep_vars)
    if args is None:
        robin = _linear_boundary_terms(eq.lhs, dep_func, indep_vars)
        if robin is not None:
            var, loc, alpha, beta, multi = robin
            return RobinCondition(
                eq,
                "boundary",
                variable=var,
                location=loc,
                derivative_multiindex=multi,
                metadata={"alpha": alpha, "beta": beta},
                kind="robin",
            )
        return None

    fixed = _fixed_slice(args, indep_vars)
    if fixed is None:
        return ConditionEquation(
            eq, "mixed", metadata={"reason": "nonconstant_slice"}, kind="mixed"
        )
    if len(fixed) == 1:
        var, loc = fixed[0]
        role = "initial" if indep_vars and var == indep_vars[-1] else "boundary"
        if role == "initial":
            kind = "profile" if sum(multi) == 0 else "initial_derivative"
            return InitialCondition(
                eq,
                role,
                variable=var,
                location=loc,
                derivative_multiindex=multi,
                kind=kind,
            )
        spatial_idx = indep_vars.index(var)
        s_order = multi[spatial_idx] if spatial_idx < len(multi) else 0
        if sum(multi) == 0:
            return DirichletCondition(
                eq,
                role,
                variable=var,
                location=loc,
                derivative_multiindex=multi,
                kind="dirichlet",
            )
        if s_order == 1 and sum(multi) == 1:
            return NeumannCondition(
                eq,
                role,
                variable=var,
                location=loc,
                derivative_multiindex=multi,
                kind="neumann",
            )
        return RobinCondition(
            eq,
            role,
            variable=var,
            location=loc,
            derivative_multiindex=multi,
            kind="robin",
        )
    if len(fixed) > 1:
        return InterfaceCondition(
            eq,
            "mixed",
            variable=fixed[0][0],
            location=fixed[0][1],
            derivative_multiindex=multi,
            metadata={"fixed_variables": tuple(fixed)},
            kind="mixed",
        )
    return ConditionEquation(eq, "unknown", derivative_multiindex=multi, kind="unknown")


def _with_role(cond: ConditionEquation, role: str) -> ConditionEquation:
    if cond.role == role:
        return cond
    if role == "initial":
        kind = "profile" if cond.derivative_order == 0 else "initial_derivative"
        return InitialCondition(
            cond.equation,
            role,
            variable=cond.variable,
            location=cond.location,
            derivative_multiindex=cond.derivative_multiindex,
            metadata=dict(cond.metadata),
            kind=kind,
        )
    if role == "boundary":
        total = cond.derivative_order
        kind = "dirichlet" if total == 0 else "neumann" if total == 1 else "robin"
        cls = (
            DirichletCondition
            if total == 0
            else NeumannCondition
            if total == 1
            else RobinCondition
        )
        return cls(
            cond.equation,
            role,
            variable=cond.variable,
            location=cond.location,
            derivative_multiindex=cond.derivative_multiindex,
            metadata=dict(cond.metadata),
            kind=kind,
        )
    return replace(cond, role=role)


def parse_conditions(
    ics=None, bcs=None, *, dep_expr=None, indep_vars: tuple[sp.Symbol, ...] = ()
) -> ConditionModel:
    dep_func = _dep_func(dep_expr)
    init: list[ConditionEquation] = []
    bc: list[ConditionEquation] = []
    mixed: list[ConditionEquation] = []
    events: list[ConditionEquation] = []
    meta: dict[str, Any] = {}

    def store(ce, hinted_role=None):
        if ce is None:
            return
        role = hinted_role or ce.role
        ce = _with_role(ce, role)
        if role == "initial":
            init.append(ce)
        elif role == "boundary":
            bc.append(ce)
        elif role == "event":
            events.append(ce)
        else:
            mixed.append(ce)

    def ingest(source, hinted_role: str | None = None):
        if source is None:
            return
        if isinstance(source, ConditionEquation):
            store(source, hinted_role)
            return
        if isinstance(source, dict):
            meta.setdefault("raw_dicts", []).append(dict(source))
            eqs = list(source.get("equations", ()) or ())
            for key in ("equation", "initial_equation"):
                if source.get(key) is not None:
                    eqs.append(source[key])
            for eq in eqs:
                if isinstance(eq, sp.Equality):
                    store(_condition_from_eq(eq, dep_func, indep_vars), hinted_role)
            return
        if isinstance(source, (list, tuple)):
            for item in source:
                ingest(item, hinted_role=hinted_role)
            return
        eq = _as_eq(source)
        if eq is not None:
            store(_condition_from_eq(eq, dep_func, indep_vars), hinted_role)

    ingest(ics, hinted_role="initial")
    ingest(bcs, hinted_role="boundary")
    time_vars = {cond.variable for cond in init if cond.variable is not None}
    if len(time_vars) == 1:
        meta["time_variable"] = next(iter(time_vars))
    return ConditionModel(
        dep_expr,
        tuple(indep_vars),
        tuple(init),
        tuple(bc),
        tuple(mixed),
        tuple(events),
        meta,
    )


def classify_condition_equation(
    cond: ConditionEquation,
    *,
    time_variable: sp.Symbol | None = None,
    spatial_variables: tuple[sp.Symbol, ...] = (),
    independent_variables: tuple[sp.Symbol, ...] = (),
) -> str:
    """Return the canonical mathematical kind of a condition."""
    if cond.kind and cond.kind not in {"initial_derivative"}:
        if cond.role == "initial" and cond.kind == "profile" and cond.derivative_order:
            pass
        else:
            return cond.kind
    if cond.role == "event":
        return "event"
    if cond.role == "mixed":
        return "mixed"
    deriv = tuple(cond.derivative_multiindex or ())
    if cond.role == "initial":
        if time_variable is not None and cond.variable == time_variable:
            if sum(deriv) == 0:
                return "profile"
            all_vars = tuple(independent_variables)
            if all_vars:
                try:
                    t_index = all_vars.index(time_variable)
                except ValueError:
                    t_index = len(deriv)
            else:
                t_index = len(spatial_variables)
            t_order = deriv[t_index] if t_index < len(deriv) else 0
            if t_order == 1 and sum(deriv) == 1:
                return "velocity"
            if t_order == 2 and sum(deriv) == 2:
                return "acceleration"
        return "initial_derivative" if sum(deriv) else "profile"
    if cond.role == "boundary":
        if cond.metadata.get("paired_location") is not None:
            return "periodic"
        spatial_idx = next(
            (i for i, v in enumerate(spatial_variables) if cond.variable == v), None
        )
        s_order = (
            deriv[spatial_idx]
            if spatial_idx is not None and spatial_idx < len(deriv)
            else 0
        )
        total = sum(deriv)
        if (
            cond.metadata.get("alpha") is not None
            or cond.metadata.get("beta") is not None
        ):
            return "robin"
        if total == 0:
            return "dirichlet"
        if s_order == 1 and total == 1:
            return "neumann"
        if s_order >= 1:
            return "robin"
        return "boundary_derivative"
    return "unknown"


def summarize_condition_model(model: ConditionModel) -> dict[str, Any]:
    indep = tuple(model.independent_variables)
    time_var = model.metadata.get("time_variable")
    if time_var is None and len(indep) >= 2:
        time_var = indep[-1]
    spatial = (
        tuple(v for v in indep if v != time_var) if time_var is not None else indep
    )
    ic_kinds = tuple(
        classify_condition_equation(
            c,
            time_variable=time_var,
            spatial_variables=spatial,
            independent_variables=indep,
        )
        for c in model.initial_conditions
    )
    bc_kinds = tuple(
        classify_condition_equation(
            c,
            time_variable=time_var,
            spatial_variables=spatial,
            independent_variables=indep,
        )
        for c in model.boundary_conditions
    )
    time_slices = tuple(
        sorted(
            {c.location for c in model.initial_conditions if c.location is not None},
            key=sp.default_sort_key,
        )
    )
    boundary_locs = tuple(
        sorted(
            {
                (str(c.variable), c.location)
                for c in model.boundary_conditions
                if c.location is not None
            },
            key=lambda t: (t[0], sp.default_sort_key(t[1])),
        )
    )
    return {
        "initial_kinds": ic_kinds,
        "boundary_kinds": bc_kinds,
        "time_slices": time_slices,
        "boundary_locations": boundary_locs,
        "has_mixed": bool(model.mixed_conditions),
        "has_events": bool(model.event_conditions),
    }


def first_constant_time_slice(model: ConditionModel):
    summary = summarize_condition_model(model)
    return summary["time_slices"][0] if summary["time_slices"] else None


def extract_equations_by_role(
    model: ConditionModel, role: str
) -> tuple[sp.Equality, ...]:
    groups = {
        "initial": model.initial_conditions,
        "boundary": model.boundary_conditions,
        "mixed": model.mixed_conditions,
        "event": model.event_conditions,
    }
    return tuple(c.equation for c in groups.get(role, ()))


def select_boundary_equations(
    model: ConditionModel | None,
    *,
    variable=None,
    location=None,
    kind: str | None = None,
):
    if model is None:
        return ()
    time_var = model.metadata.get("time_variable")
    if time_var is None and len(model.independent_variables) >= 2:
        time_var = model.independent_variables[-1]
    spatial_vars = (
        tuple(v for v in model.independent_variables if v != time_var)
        if time_var is not None
        else tuple(model.independent_variables)
    )
    out = []
    for cond in model.boundary_conditions:
        if variable is not None and cond.variable != variable:
            continue
        if location is not None and cond.location is not None:
            try:
                if sp.simplify(cond.location - location) != 0:
                    continue
            except Exception:
                if cond.location != location:
                    continue
        if (
            kind is not None
            and classify_condition_equation(
                cond,
                time_variable=time_var,
                spatial_variables=spatial_vars,
                independent_variables=tuple(model.independent_variables),
            )
            != kind
        ):
            continue
        out.append(cond.equation)
    return tuple(out)


__all__ = [
    "ConditionEquation",
    "InitialCondition",
    "DirichletCondition",
    "NeumannCondition",
    "RobinCondition",
    "PeriodicCondition",
    "InterfaceCondition",
    "AsymptoticCondition",
    "ConditionModel",
    "parse_conditions",
    "classify_condition_equation",
    "summarize_condition_model",
    "first_constant_time_slice",
    "extract_equations_by_role",
    "select_boundary_equations",
]
