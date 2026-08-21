from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class DomainGeometry:
    kind: str
    coordinates: tuple[sp.Symbol, ...]
    extents: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntervalDomain(DomainGeometry):
    pass


@dataclass(frozen=True)
class RectangleDomain(DomainGeometry):
    pass


@dataclass(frozen=True)
class DiskDomain(DomainGeometry):
    pass


@dataclass(frozen=True)
class HalfLineDomain(DomainGeometry):
    pass


@dataclass(frozen=True)
class FullLineDomain(DomainGeometry):
    pass


@dataclass(frozen=True)
class PolarAnnulusDomain(DomainGeometry):
    pass


@dataclass(frozen=True)
class HalfPlaneDomain(DomainGeometry):
    pass


@dataclass(frozen=True)
class HalfSpaceDomain(DomainGeometry):
    pass


def _sorted_locs(conds, var):
    vals = []
    for c in conds:
        if c.variable == var and c.location is not None:
            vals.append(c.location)
    try:
        return tuple(sorted(set(vals), key=sp.default_sort_key))
    except Exception:
        return tuple(dict.fromkeys(vals))


def infer_domain_geometry(
    *,
    indep_vars: tuple[sp.Symbol, ...],
    bcs=None,
    condition_model=None,
    domain=None,
    assumptions=True,
) -> DomainGeometry:
    vars_ = tuple(indep_vars)
    # Explicit domains take precedence.  Accept native DomainGeometry, SymPy
    # Interval, and simple ProductSet descriptions.
    if isinstance(domain, DomainGeometry):
        return domain
    if isinstance(domain, sp.Interval) and vars_:
        a, b = domain.start, domain.end
        if a is -sp.oo or a == -sp.oo:
            if b is sp.oo or b == sp.oo:
                return FullLineDomain(
                    "full_line", vars_[:1], {str(vars_[0]): (a, b)}, {"source": "explicit_domain"}
                )
        if b is sp.oo or b == sp.oo:
            return HalfLineDomain(
                "half_line", vars_[:1], {str(vars_[0]): (a, b)}, {"source": "explicit_domain"}
            )
        return IntervalDomain(
            "interval", vars_[:1], {str(vars_[0]): (a, b)}, {"source": "explicit_domain"}
        )
    if isinstance(domain, sp.ProductSet) and len(vars_) >= len(domain.args):
        intervals = list(domain.args)
        if all(isinstance(item, sp.Interval) for item in intervals):
            ext = {str(v): (item.start, item.end) for v, item in zip(vars_, intervals, strict=True)}
            bounded = [item for item in intervals if item.start != -sp.oo and item.end != sp.oo]
            if len(intervals) == 2 and len(bounded) == 2:
                return RectangleDomain("rectangle", vars_[:2], ext, {"source": "explicit_domain"})
            return DomainGeometry(
                "product", vars_[: len(intervals)], ext, {"source": "explicit_domain"}
            )
    if isinstance(bcs, dict):
        btype = bcs.get("type")
        if btype in {"dirichlet_rectangle", "rectangle"} and len(vars_) >= 2:
            return RectangleDomain(
                "rectangle",
                vars_[:2],
                {
                    "x": (bcs.get("x0", 0), bcs.get("x1", sp.pi)),
                    "y": (bcs.get("y0", 0), bcs.get("y1", sp.pi)),
                },
                {"source": "bc_dict"},
            )
        if (
            btype in {"half_plane", "dirichlet_half_plane", "neumann_half_plane"}
            and len(vars_) >= 2
        ):
            return HalfPlaneDomain(
                "half_plane", vars_[:2], {str(vars_[1]): (0, sp.oo)}, {"source": "bc_dict"}
            )
        if (
            btype in {"half_space", "dirichlet_half_space", "neumann_half_space"}
            and len(vars_) >= 3
        ):
            return HalfSpaceDomain(
                "half_space", vars_[:3], {str(vars_[2]): (0, sp.oo)}, {"source": "bc_dict"}
            )
        if btype in {"strip", "infinite_strip"} and len(vars_) >= 2:
            return RectangleDomain(
                "strip",
                vars_[:2],
                {
                    str(vars_[1]): (bcs.get("y0", 0), bcs.get("y1", sp.Symbol("a", positive=True))),
                    str(vars_[0]): (-sp.oo, sp.oo),
                },
                {"source": "bc_dict"},
            )
        if btype in {"semi_infinite_strip", "semistrip"} and len(vars_) >= 2:
            return RectangleDomain(
                "semi_infinite_strip",
                vars_[:2],
                {
                    str(vars_[0]): (0, sp.oo),
                    str(vars_[1]): (bcs.get("y0", 0), bcs.get("y1", sp.Symbol("a", positive=True))),
                },
                {"source": "bc_dict"},
            )
        if btype == "quadrant" and len(vars_) >= 2:
            return DomainGeometry(
                "quadrant",
                vars_[:2],
                {str(vars_[0]): (0, sp.oo), str(vars_[1]): (0, sp.oo)},
                {"source": "bc_dict"},
            )
        if (
            btype
            in {
                "dirichlet_half_line",
                "neumann_half_line",
                "half_line_dirichlet",
                "half_line_neumann",
            }
            and vars_
        ):
            return HalfLineDomain(
                "half_line",
                vars_[:1],
                {str(vars_[0]): (bcs.get("x0", 0), sp.oo)},
                {"source": "bc_dict"},
            )
        if (
            btype
            in {"dirichlet_homogeneous_interval", "neumann_homogeneous_interval", "robin_interval"}
            and vars_
        ):
            return IntervalDomain(
                "interval",
                vars_[:1],
                {str(vars_[0]): (bcs.get("x0", 0), bcs.get("x1", bcs.get("length", sp.pi)))},
                {"source": "bc_dict"},
            )
        if btype == "disk" and len(vars_) >= 2:
            return DiskDomain(
                "disk", vars_[:2], {"r": (0, bcs.get("radius", 1))}, {"source": "bc_dict"}
            )
    if condition_model is not None and len(vars_) >= 1:
        has_time = len(vars_) >= 2 and any(
            c.role == "initial" for c in condition_model.initial_conditions
        )
        # Boundary-only 1+1D problems commonly omit initial data.  If all
        # supplied boundaries constrain the first variable while the second
        # remains free, treat the latter as the evolution coordinate so the
        # spatial interval can still be inferred.
        if not has_time and len(vars_) == 2 and condition_model.boundary_conditions:
            first_locs = _sorted_locs(condition_model.boundary_conditions, vars_[0])
            second_locs = _sorted_locs(condition_model.boundary_conditions, vars_[1])
            if len(first_locs) >= 2 and len(second_locs) == 0:
                has_time = True
        time_var = (condition_model.metadata or {}).get("time_variable") if has_time else None
        if has_time and time_var is None and len(vars_) >= 2:
            time_var = vars_[-1]
        spatial_vars = tuple(v for v in vars_ if v != time_var) if has_time else vars_
        loc_map = {sv: _sorted_locs(condition_model.boundary_conditions, sv) for sv in spatial_vars}
        extents = {str(sv): (vals[0], vals[-1]) for sv, vals in loc_map.items() if len(vals) >= 2}
        if has_time and len(spatial_vars) == 1 and not condition_model.boundary_conditions:
            sv = spatial_vars[0]
            return FullLineDomain(
                "full_line",
                (sv,),
                {str(sv): (-sp.oo, sp.oo)},
                {"source": "initial_data_no_spatial_boundary"},
            )
        if has_time and len(spatial_vars) == 1 and condition_model.boundary_conditions:
            sv = spatial_vars[0]
            vals = loc_map.get(sv, ())
            # A single spatial boundary plus an evolution initial condition is
            # a half-line only when orientation is supported by symbol
            # assumptions or the boundary is explicitly at the conventional
            # origin.  Record that this is an inference, not explicit input.
            if len(vals) == 1 and (
                getattr(sv, "is_nonnegative", None) is True
                or getattr(sv, "is_positive", None) is True
                or vals[0] == 0
            ):
                return HalfLineDomain(
                    "half_line",
                    (sv,),
                    {str(sv): (vals[0], sp.oo)},
                    {"source": "conditions", "inferred_orientation": "right"},
                )
        if len(spatial_vars) == 1 and len(extents) == 1:
            sv = spatial_vars[0]
            return IntervalDomain(
                "interval", (sv,), {"x": extents[str(sv)]}, {"source": "conditions"}
            )
        if len(spatial_vars) == 2:
            xvar, yvar = spatial_vars
            xlocs, ylocs = loc_map.get(xvar, ()), loc_map.get(yvar, ())
            if len(xlocs) >= 2 and len(ylocs) >= 2:
                x0, x1 = xlocs[0], xlocs[-1]
                y0, y1 = ylocs[0], ylocs[-1]
                x_unbounded = x0 == 0 and x1 == sp.oo
                if x_unbounded:
                    return RectangleDomain(
                        "semi_infinite_strip",
                        spatial_vars,
                        {str(xvar): (x0, x1), str(yvar): (y0, y1)},
                        {"source": "conditions"},
                    )
                return RectangleDomain(
                    "rectangle",
                    spatial_vars,
                    {str(xvar): (x0, x1), str(yvar): (y0, y1)},
                    {"source": "conditions"},
                )
            if len(ylocs) >= 2 and len(xlocs) == 0:
                return RectangleDomain(
                    "strip",
                    spatial_vars,
                    {str(xvar): (-sp.oo, sp.oo), str(yvar): (ylocs[0], ylocs[-1])},
                    {"source": "conditions"},
                )
            if len(xlocs) == 1 and len(ylocs) == 1 and xlocs[0] == 0 and ylocs[0] == 0:
                return DomainGeometry(
                    "quadrant",
                    spatial_vars,
                    {str(xvar): (0, sp.oo), str(yvar): (0, sp.oo)},
                    {"source": "conditions"},
                )
            if len(ylocs) == 1 and ylocs[0] == 0 and len(xlocs) == 0:
                return HalfPlaneDomain(
                    "half_plane", spatial_vars, {str(yvar): (0, sp.oo)}, {"source": "conditions"}
                )
        if len(spatial_vars) == 3:
            zvar = spatial_vars[-1]
            zlocs = loc_map.get(zvar, ())
            if len(zlocs) == 1 and zlocs[0] == 0:
                return HalfSpaceDomain(
                    "half_space", spatial_vars, {str(zvar): (0, sp.oo)}, {"source": "conditions"}
                )
    if len(vars_) >= 2:
        return DomainGeometry("unspecified_spacetime", vars_)
    if len(vars_) == 1:
        return DomainGeometry("unspecified", vars_, metadata={"source": "unspecified"})
    return DomainGeometry("unspecified", vars_)


def summarize_domain_geometry(domain: DomainGeometry | None) -> dict[str, Any]:
    if domain is None:
        return {"kind": "unspecified", "axes": (), "bounded_axes": (), "is_spacetime": False}
    bounded_axes = tuple(
        k
        for k, v in (domain.extents or {}).items()
        if isinstance(v, tuple) and len(v) == 2 and all(val not in (-sp.oo, sp.oo) for val in v)
    )
    return {
        "kind": domain.kind,
        "axes": tuple(str(v) for v in domain.coordinates),
        "bounded_axes": bounded_axes,
        "is_spacetime": "spacetime" in domain.kind,
        "extents": dict(domain.extents or {}),
    }


__all__ = [
    "DomainGeometry",
    "IntervalDomain",
    "RectangleDomain",
    "DiskDomain",
    "HalfLineDomain",
    "FullLineDomain",
    "PolarAnnulusDomain",
    "HalfPlaneDomain",
    "HalfSpaceDomain",
    "infer_domain_geometry",
    "summarize_domain_geometry",
]
