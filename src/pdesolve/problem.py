from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from types import MappingProxyType

import sympy as sp

from .classification import preprocess_pde_problem
from .results import CanonicalPDERepresentation
from .condition_analysis import analyze_conditions
from .boundary_model import build_boundary_model
from .hyperbolic_system import extract_canonical_linear_system_form
from .kernels import build_kernel_method_plan


@dataclass(frozen=True)
class PDEProblem:
    equation: sp.Equality
    dep_function: sp.Expr
    indep_vars: tuple[sp.Symbol, ...]
    ics: Any = None
    bcs: Any = None
    domain: Any = None
    assumptions: Any = True
    profile: Any = None
    normalized_data: Any = None
    details: dict[str, Any] = field(default_factory=dict)
    canonical_representation: CanonicalPDERepresentation | None = None

    def __post_init__(self):
        object.__setattr__(self, "details", MappingProxyType(dict(self.details or {})))


def build_pde_problem(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    domain=None,
    assumptions=True,
):
    from . import classical_methods as classical_mod

    uexpr, vars_ = classical_mod._dep_and_vars(dep_expr_or_func, indep_vars)
    can = classical_mod.canonicalize_pde_problem(
        eq_or_expr, uexpr, vars_, assumptions=assumptions
    )
    profile = preprocess_pde_problem(
        can.equation, uexpr, vars_, assumptions=assumptions
    )
    try:
        normalized_data = classical_mod.normalize_problem_data(
            ics=ics,
            bcs=bcs,
            indep_vars=vars_,
            assumptions=assumptions,
            dep_expr=uexpr,
        )
    except (ValueError, NotImplementedError):
        normalized_data = None
    from .recognition import build_canonical_representation

    canonical_representation = (
        build_canonical_representation(
            profile,
            ics=ics,
            bcs=bcs,
            normalized_data=normalized_data,
            dep_expr=uexpr,
            domain=domain,
        )
        if profile is not None
        else None
    )
    condition_model = (
        getattr(canonical_representation, "details", {}).get("condition_model")
        if canonical_representation
        else None
    )
    domain_geometry = (
        getattr(canonical_representation, "details", {}).get("domain_geometry")
        if canonical_representation
        else None
    )
    condition_analysis = (
        analyze_conditions(
            condition_model,
            domain_geometry,
            pde_order=getattr(profile, "order", None),
            dependent_variables=(uexpr,),
        )
        if condition_model is not None
        else None
    )
    boundary_model = (
        build_boundary_model(domain_geometry, condition_model)
        if condition_model is not None
        else None
    )
    transform_plan = (
        getattr(canonical_representation, "details", {}).get("transform_plan")
        if canonical_representation
        else None
    )
    separation_plan = (
        getattr(canonical_representation, "details", {}).get("separation_plan")
        if canonical_representation
        else None
    )
    kernel_plan = None
    problem = PDEProblem(
        equation=can.equation,
        dep_function=uexpr,
        indep_vars=tuple(vars_),
        ics=ics,
        bcs=bcs,
        domain=domain,
        assumptions=assumptions,
        profile=profile,
        normalized_data=normalized_data,
        details={
            "canonicalization": can,
            "recognitions": profile.details.get("recognitions", ())
            if hasattr(profile, "details")
            else (),
            "condition_model": condition_model,
            "domain_geometry": domain_geometry,
            "condition_analysis": condition_analysis,
            "boundary_model": boundary_model,
            "transform_plan": transform_plan,
            "separation_plan": separation_plan,
            "kernel_plan": None,
        },
        canonical_representation=canonical_representation,
    )
    try:
        kernel_plan = build_kernel_method_plan(problem)
    except (ValueError, NotImplementedError):
        kernel_plan = None
    if kernel_plan is not None:
        new_details = dict(problem.details)
        new_details["kernel_plan"] = kernel_plan
        problem = replace(problem, details=new_details)
    return problem


def build_system_pde_problem(
    eqns, dep_exprs_or_funcs, indep_vars, *, ics=None, assumptions=True
):
    vars_ = tuple(indep_vars)
    unknowns = tuple(dep_exprs_or_funcs)
    canonical_system = None
    try:
        canonical_system = (
            extract_canonical_linear_system_form(eqns, unknowns, vars_[:2])
            if len(vars_) >= 2
            else None
        )
    except (ValueError, NotImplementedError):
        canonical_system = None
    problem = PDEProblem(
        equation=tuple(eqns),
        dep_function=tuple(unknowns),
        indep_vars=vars_,
        ics=ics,
        bcs=None,
        domain=None,
        assumptions=assumptions,
        profile=None,
        normalized_data=None,
        details={
            "raw_equations": tuple(eqns),
            "raw_unknowns": tuple(unknowns),
            "raw_ics": tuple(ics or ()),
            "canonical_system": canonical_system,
        },
        canonical_representation=None,
    )

    return problem
