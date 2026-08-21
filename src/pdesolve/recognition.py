from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from .boundary_model import build_boundary_model
from .condition_analysis import analyze_conditions
from .conditions import parse_conditions, summarize_condition_model
from .domains import infer_domain_geometry, summarize_domain_geometry
from .first_order_framework import canonicalize_first_order_nonlinear_pde
from .results import CanonicalPDERepresentation, PDEProblemProfile
from .separation_framework import build_separable_geometry_plan
from .transform_framework import build_transform_method_plan


@dataclass(frozen=True)
class PDERecognitionRecord:
    family: str
    recognized: bool
    solver_hint: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _linearity_from_zero_expr(zero: sp.Expr, dep_expr: sp.Expr) -> str:
    derivs = [
        d for d in zero.atoms(sp.Derivative) if getattr(d.expr, "func", None) == dep_expr.func
    ]
    atoms = derivs + ([dep_expr] if zero.has(dep_expr) else [])
    if not atoms:
        return "independent"
    try:
        poly = sp.Poly(sp.expand(zero), *atoms)
        if poly.total_degree() <= 1:
            coeffs = [sp.expand(sp.diff(zero, a)) for a in atoms]
            nonlinear_coeff = any(
                any(sym in c.free_symbols for sym in dep_expr.args) or c.has(dep_expr)
                for c in coeffs
            )
            return "quasilinear" if nonlinear_coeff else "linear"
    except Exception:
        pass
    # conservative fallback
    nonlinear = False
    for a in atoms:
        try:
            if sp.diff(zero, a, 2) != 0:
                nonlinear = True
                break
        except Exception:
            nonlinear = True
            break
    return "fully_nonlinear" if nonlinear else "quasilinear"


def build_canonical_representation(
    profile: PDEProblemProfile,
    *,
    ics=None,
    bcs=None,
    normalized_data=None,
    dep_expr=None,
    domain=None,
) -> CanonicalPDERepresentation:
    normalized = profile.normalized_equation
    dep = profile.dep_function
    vars_ = profile.indep_vars
    zero = profile.zero_expression
    transform_tags: list[str] = []
    if profile.canonical_family in {"heat_like", "wave_like"}:
        transform_tags.extend(["separation_of_variables", "integral_transform"])
    if profile.conservation_law is not None:
        transform_tags.append("weak_solution")
    if profile.characteristic_data is not None:
        transform_tags.append("characteristics")
    if profile.details.get("constant_coefficient_profile") is not None:
        transform_tags.append("constant_coefficients")

    principal = profile.principal_solved_form
    coeff_dep: list[str] = []
    if principal is not None:
        coeff_dep.append("solved_principal_part")
    if profile.first_order_linear is not None and getattr(
        profile.first_order_linear, "is_constant_coefficient", False
    ):
        coeff_dep.append("constant")
    elif profile.first_order_linear is not None:
        coeff_dep.append("variable")
    if profile.conservation_law is not None:
        coeff_dep.append("depends_on_u")

    recognized = []
    if profile.canonical_family:
        recognized.append(profile.canonical_family)
    if profile.conservation_law is not None:
        recognized.append("scalar_conservation_law")

    ic_meta = {"provided": ics is not None, "count": 0}
    bc_meta = {"provided": bcs is not None, "count": 0}
    time_slice = {}
    domain_meta = {"assumptions": True, "explicit": domain is not None}
    geom_meta = {"family": profile.canonical_family}
    if normalized_data is not None:
        try:
            ic_meta["count"] = len(getattr(normalized_data, "initial_conditions", ()) or ())
            bc_meta["count"] = len(getattr(normalized_data, "boundary_conditions", ()) or ())
            dom = getattr(normalized_data, "domain", None)
            if dom is not None:
                domain_meta["geometry"] = getattr(dom, "geometry", None)
                domain_meta["spatial_interval"] = getattr(dom, "spatial_interval", None)
                domain_meta["time_interval"] = getattr(dom, "time_interval", None)
                if getattr(dom, "metadata", None):
                    geom = dom.metadata.get("geometry")
                    if geom is not None:
                        geom_meta["kind"] = getattr(geom, "kind", None)
                        geom_meta["extents"] = getattr(geom, "extents", {})
        except Exception:
            pass
    cmodel = parse_conditions(ics, bcs, dep_expr=dep_expr or dep, indep_vars=tuple(vars_))
    if cmodel.initial_conditions:
        ic_meta["count"] = len(cmodel.initial_conditions)
        time_slice["constant_time_values"] = tuple(
            sorted(
                {c.location for c in cmodel.initial_conditions if c.location is not None},
                key=sp.default_sort_key,
            )
        )
    if cmodel.boundary_conditions:
        bc_meta["count"] = len(cmodel.boundary_conditions)
        geom_meta["boundary_variables"] = tuple(
            sorted({str(c.variable) for c in cmodel.boundary_conditions if c.variable is not None})
        )

    geom = infer_domain_geometry(
        indep_vars=tuple(vars_), bcs=bcs, condition_model=cmodel, domain=domain
    )
    condition_report = analyze_conditions(
        cmodel, geom, pde_order=profile.order, dependent_variables=(dep,)
    )
    sep_plan = build_separable_geometry_plan(geom, cmodel, family=profile.canonical_family)
    boundary_model = build_boundary_model(geom, cmodel)
    first_order_model = None
    if profile.order == 1 and len(vars_) >= 2:
        try:
            first_order_model = canonicalize_first_order_nonlinear_pde(normalized, dep, vars_)
        except Exception:
            first_order_model = None
    weak_flags = tuple(
        sorted(
            {
                "riemann_data" if isinstance(ics, dict) and "riemann_data" in ics else None,
                "piecewise" if zero.has(sp.Piecewise) else None,
                "multi_interface"
                if zero.has(sp.Piecewise) and len(list(zero.atoms(sp.Piecewise))) > 0
                else None,
            }
            - {None}
        )
    )
    return CanonicalPDERepresentation(
        dependent_variables=(dep,),
        independent_variables=tuple(vars_),
        normalized_equations=(normalized,),
        order=profile.order,
        linearity=_linearity_from_zero_expr(zero, dep),
        principal_part=principal,
        coefficient_dependence=tuple(coeff_dep),
        principal_multiindex=profile.details.get("principal_multiindex"),
        ic_metadata=ic_meta,
        bc_metadata=bc_meta,
        domain_metadata=domain_meta,
        geometry_metadata=geom_meta,
        time_slice_metadata=time_slice,
        weak_solution_flags=weak_flags,
        transformability_tags=tuple(sorted(set(transform_tags))),
        recognized_tags=tuple(sorted(set(recognized))),
        details={
            "profile_details": profile.details,
            "condition_model": cmodel,
            "condition_summary": summarize_condition_model(cmodel),
            "domain_geometry": geom,
            "domain_summary": summarize_domain_geometry(geom),
            "condition_analysis": condition_report,
            "separation_plan": sep_plan,
            "transform_plan": build_transform_method_plan(
                CanonicalPDERepresentation(
                    dependent_variables=(dep,),
                    independent_variables=tuple(vars_),
                    normalized_equations=(normalized,),
                    order=profile.order,
                    linearity=_linearity_from_zero_expr(zero, dep),
                    principal_part=principal,
                    coefficient_dependence=tuple(coeff_dep),
                    principal_multiindex=profile.details.get("principal_multiindex"),
                    ic_metadata=ic_meta,
                    bc_metadata=bc_meta,
                    domain_metadata=domain_meta,
                    geometry_metadata=geom_meta,
                    time_slice_metadata=time_slice,
                    weak_solution_flags=(),
                    transformability_tags=tuple(sorted(set(transform_tags))),
                    recognized_tags=tuple(sorted(set(recognized))),
                    details={},
                ),
                geom,
                cmodel,
            ),
            "boundary_model": boundary_model,
            "first_order_model": first_order_model,
        },
    )


def recognize_pde_structure(
    profile: PDEProblemProfile, *, canonical: CanonicalPDERepresentation | None = None
) -> tuple[PDERecognitionRecord, ...]:
    recs: list[PDERecognitionRecord] = []
    can = canonical
    if can is None:
        can = build_canonical_representation(profile)
    if profile.conservation_law is not None:
        recs.append(
            PDERecognitionRecord(
                "scalar_conservation_law",
                True,
                "conservation_law",
                ("weak_solution", "riemann_ready"),
                {"canonical_family": profile.canonical_family},
            )
        )
    if profile.first_order_linear is not None:
        recs.append(
            PDERecognitionRecord(
                "first_order_linear",
                True,
                "first_order",
                ("characteristics",),
                {
                    "constant_coefficient": getattr(
                        profile.first_order_linear, "is_constant_coefficient", None
                    )
                },
            )
        )
    if profile.characteristic_data is not None and profile.first_order_linear is None:
        recs.append(
            PDERecognitionRecord(
                "quasilinear_scalar_ivp", True, "quasilinear_implicit", ("characteristics",), {}
            )
        )
    fam = profile.canonical_family
    if fam == "heat_like":
        recs.append(
            PDERecognitionRecord(
                "heat_like",
                True,
                "heat_dirichlet_series",
                ("separation_of_variables", "integral_transform"),
                {},
            )
        )
    elif fam == "wave_like":
        recs.append(
            PDERecognitionRecord(
                "wave_like", True, "wave_dalembert", ("dAlembert", "separation_of_variables"), {}
            )
        )
    from .complete_integral_helpers import recognize_generalized_clairaut_pde

    try:
        clair = recognize_generalized_clairaut_pde(
            profile.normalized_equation, profile.dep_function, profile.indep_vars
        )
        if clair.recognized:
            recs.append(
                PDERecognitionRecord(
                    "generalized_clairaut",
                    True,
                    "generalized_clairaut_complete_integral",
                    ("complete_integral", "charpit"),
                    {"phi": clair.phi},
                )
            )
    except Exception:
        pass
    return tuple(recs)


__all__ = ["PDERecognitionRecord", "build_canonical_representation", "recognize_pde_structure"]
