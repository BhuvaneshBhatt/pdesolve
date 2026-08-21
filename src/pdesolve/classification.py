from __future__ import annotations

import sympy as sp
from collections import OrderedDict
from dataclasses import replace
from threading import RLock

from .results import PDEProblemProfile, PDESolverMethodCandidate, PDESolutionPlan
from .recognition import build_canonical_representation, recognize_pde_structure
from .conditions import parse_conditions, summarize_condition_model
from .domains import infer_domain_geometry, summarize_domain_geometry
from .condition_analysis import analyze_conditions
from .separation_framework import build_separable_geometry_plan
from .transform_framework import build_transform_method_plan
from .kernels import build_kernel_method_plan
from .boundary_model import build_boundary_model


def _cache_key(*objs):
    parts = []
    for obj in objs:
        try:
            parts.append(sp.srepr(obj))
        except Exception:
            parts.append(repr(obj))
    return tuple(parts)


_PREPROCESS_CACHE: OrderedDict[tuple, PDEProblemProfile] = OrderedDict()
_PREPROCESS_CACHE_MAXSIZE = 256
_PREPROCESS_CACHE_LOCK = RLock()


def clear_preprocess_cache() -> None:
    """Clear cached PDE preprocessing profiles."""
    with _PREPROCESS_CACHE_LOCK:
        _PREPROCESS_CACHE.clear()


def _profile_copy(profile: PDEProblemProfile) -> PDEProblemProfile:
    # Cached profiles are never returned directly: callers may enrich the
    # top-level details mapping while planning a solve.  A shallow mapping
    # copy isolates those changes without duplicating large SymPy objects.
    return replace(profile, details=dict(profile.details))


def _refine_family(normalized, uexpr, vars_, fallback):
    if len(vars_) == 2:
        x, t = vars_
        zero = sp.expand(
            (normalized.lhs - normalized.rhs)
            if isinstance(normalized, sp.Equality)
            else normalized
        )
        ut = sp.diff(uexpr, t)
        uxx = sp.diff(uexpr, x, 2)
        utt = sp.diff(uexpr, t, 2)
        try:
            a_t = sp.expand(sp.diff(zero, ut))
            a_xx = sp.expand(sp.diff(zero, uxx))
            rem = sp.expand(zero - a_t * ut - a_xx * uxx)
            if a_t != 0 and a_xx != 0 and rem == 0:
                return "heat_like"
        except Exception:
            pass
        try:
            a_tt = sp.expand(sp.diff(zero, utt))
            a_xx = sp.expand(sp.diff(zero, uxx))
            rem = sp.expand(zero - a_tt * utt - a_xx * uxx)
            if a_tt != 0 and a_xx != 0 and rem == 0:
                return "wave_like"
        except Exception:
            pass
    return fallback


def preprocess_pde_problem(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    assumptions=True,
    max_principal_order=3,
) -> PDEProblemProfile:
    key = _cache_key(
        eq_or_expr, dep_expr_or_func, indep_vars, assumptions, max_principal_order
    )
    with _PREPROCESS_CACHE_LOCK:
        cached = _PREPROCESS_CACHE.get(key)
        if cached is not None:
            _PREPROCESS_CACHE.move_to_end(key)
            return _profile_copy(cached)
    from ._classical_shared import _dep_and_vars, _as_zero_expr
    from .classical_methods import (
        canonicalize_pde_problem,
        _infer_pde_order,
        characteristic_form_first_order_2vars,
        detect_first_order_linear_form_2vars,
        classify_linear_second_order_pde,
        detect_linear_constant_coefficient_pde,
    )
    from .family_recognizers import (
        recognize_pde_family,
        detect_scalar_conservation_law_family,
    )

    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    can = canonicalize_pde_problem(eq_or_expr, uexpr, vars_, assumptions=assumptions)
    normalized = can.equation
    zero = _as_zero_expr(normalized)
    order = _infer_pde_order(zero, uexpr)
    details: dict[str, object] = {"canonicalization": can}
    first_char = None
    first_lin = None
    second_cls = None
    conslaw = None
    canonical_family = None
    solved_form = None
    if order == 1 and len(vars_) == 2:
        try:
            first_char = characteristic_form_first_order_2vars(normalized, uexpr, vars_)
        except Exception as exc:
            details["characteristic_data_error"] = str(exc)
        try:
            first_lin = detect_first_order_linear_form_2vars(normalized, uexpr, vars_)
        except Exception as exc:
            details["first_order_linear_error"] = str(exc)
        try:
            conslaw = detect_scalar_conservation_law_family(normalized, uexpr, vars_)
        except Exception as exc:
            details["conservation_law_error"] = str(exc)
    if order == 2:
        try:
            second_cls = classify_linear_second_order_pde(
                normalized, uexpr, vars_, assumptions=assumptions
            )
        except Exception as exc:
            details["second_order_linear_error"] = str(exc)

    try:
        ccpde = detect_linear_constant_coefficient_pde(normalized, uexpr, vars_)
        details["constant_coefficient_profile"] = ccpde
    except Exception as exc:
        ccpde = None
        details["constant_coefficient_error"] = str(exc)
    try:
        fam = recognize_pde_family(normalized, uexpr, vars_, assumptions=assumptions)
        if fam is not None:
            canonical_family = _refine_family(normalized, uexpr, vars_, fam.family)
            details["canonical_family"] = canonical_family
            details["family_recognition"] = fam
    except Exception as exc:
        details["canonical_family_error"] = str(exc)
    try:
        from .pde import (
            build_scalar_jet_equation_from_sympy_pde,
            build_scalar_general_solved_pde_from_equation,
        )

        jet, pde = build_scalar_jet_equation_from_sympy_pde(
            vars_,
            uexpr.func,
            normalized,
            max_order=max(max_principal_order, order or 1),
            dep_name=getattr(uexpr.func, "__name__", "u"),
        )
        solved_form, info = build_scalar_general_solved_pde_from_equation(
            jet, pde, max_principal_order=max_principal_order
        )
        details["principal_multiindex"] = getattr(info, "principal_multiindex", None)
    except Exception as exc:
        details["principal_solved_form_error"] = str(exc)
    profile = PDEProblemProfile(
        tuple(vars_),
        uexpr,
        normalized,
        zero,
        order,
        solved_form,
        first_char,
        first_lin,
        second_cls,
        canonical_family,
        conslaw,
        details,
    )
    canonical_representation = build_canonical_representation(profile)
    recognitions = recognize_pde_structure(profile, canonical=canonical_representation)
    details["canonical_representation"] = canonical_representation
    details["recognitions"] = recognitions
    result = PDEProblemProfile(
        profile.indep_vars,
        profile.dep_function,
        profile.normalized_equation,
        profile.zero_expression,
        profile.order,
        profile.principal_solved_form,
        profile.characteristic_data,
        profile.first_order_linear,
        profile.second_order_class,
        profile.canonical_family,
        profile.conservation_law,
        details,
    )
    with _PREPROCESS_CACHE_LOCK:
        _PREPROCESS_CACHE[key] = result
        _PREPROCESS_CACHE.move_to_end(key)
        while len(_PREPROCESS_CACHE) > _PREPROCESS_CACHE_MAXSIZE:
            _PREPROCESS_CACHE.popitem(last=False)
    return _profile_copy(result)


def _normalized_problem_geometry(
    indep_vars, ics=None, bcs=None, *, dep_expr=None, canonical=None, domain=None
):
    cmodel = None
    geom = None
    if canonical is not None and ics is None and bcs is None:
        cmodel = (
            canonical.details.get("condition_model")
            if getattr(canonical, "details", None)
            else None
        )
        geom = (
            canonical.details.get("domain_geometry")
            if getattr(canonical, "details", None)
            else None
        )
    if cmodel is None:
        cmodel = parse_conditions(
            ics, bcs, dep_expr=dep_expr, indep_vars=tuple(indep_vars)
        )
    if geom is None:
        geom = infer_domain_geometry(
            indep_vars=tuple(indep_vars), bcs=bcs, condition_model=cmodel, domain=domain
        )
    csummary = summarize_condition_model(cmodel)
    dsummary = summarize_domain_geometry(geom)
    condition_report = analyze_conditions(
        cmodel, geom, dependent_variables=(dep_expr,) if dep_expr is not None else ()
    )
    sep_plan = build_separable_geometry_plan(geom, cmodel)
    transform_plan = build_transform_method_plan(canonical, geom, cmodel)
    boundary_model = build_boundary_model(geom, cmodel)
    kernel_plan = None
    try:
        from .problem import PDEProblem

        tmp_problem = PDEProblem(
            equation=canonical.normalized_equations[0]
            if canonical and getattr(canonical, "normalized_equations", None)
            else dep_expr,
            dep_function=dep_expr,
            indep_vars=tuple(indep_vars),
            ics=ics,
            bcs=bcs,
            assumptions=True,
            profile=None,
            normalized_data=None,
            details={"condition_model": cmodel, "domain_geometry": geom},
            canonical_representation=canonical,
        )
        kernel_plan = build_kernel_method_plan(tmp_problem)
    except Exception:
        kernel_plan = None
    return (
        dsummary.get("kind", "unspecified"),
        set(csummary.get("initial_kinds", ())),
        set(csummary.get("boundary_kinds", ())),
        cmodel,
        geom,
        condition_report,
        sep_plan,
        transform_plan,
        boundary_model,
        kernel_plan,
    )


def rank_pde_solution_methods(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    domain=None,
    assumptions=True,
    prefer_transform=False,
    prefer_separation=False,
    prefer_symmetry=False,
):
    profile = preprocess_pde_problem(
        eq_or_expr, dep_expr_or_func, indep_vars, assumptions=assumptions
    )
    canonical = profile.details.get("canonical_representation")
    (
        geom,
        ic_kinds,
        bc_kinds,
        _cmodel,
        _dgeom,
        condition_analysis,
        sep_plan,
        transform_plan,
        _boundary_model,
        kernel_plan,
    ) = _normalized_problem_geometry(
        profile.indep_vars,
        ics=ics,
        bcs=bcs,
        dep_expr=profile.dep_function,
        canonical=canonical,
        domain=domain,
    )
    fam = profile.canonical_family
    has_profile_data = "profile" in ic_kinds or (
        isinstance(ics, dict) and ics.get("initial_profile") is not None
    )
    bc_type = bcs.get("type") if isinstance(bcs, dict) else None
    cands: list[PDESolverMethodCandidate] = []

    def add(method, score, *reasons, **details):
        cands.append(
            PDESolverMethodCandidate(
                method, int(score), tuple(str(r) for r in reasons), details
            )
        )

    if condition_analysis is not None and condition_analysis.issues:
        for issue in condition_analysis.issues:
            if issue.severity == "warning":
                add("condition_check", 18, issue.message, **issue.metadata)
    if sep_plan is not None and sep_plan.geometry_kind in {
        "interval",
        "rectangle",
        "disk",
        "polar_annulus",
    }:
        add(
            "separation_framework",
            68 if not prefer_separation else 86,
            f"structured separable geometry: {sep_plan.geometry_kind}",
            eigenbasis=sep_plan.eigenbasis,
            boundary_family=sep_plan.boundary_family,
        )
    if transform_plan is not None:
        score = 90 if prefer_transform else 77
        add(
            "structured_transform",
            score,
            f"structured transform plan: {transform_plan.transform_family}",
            domain=transform_plan.domain,
            transform_family=transform_plan.transform_family,
        )
    if kernel_plan is not None:
        recognition = (getattr(kernel_plan, "metadata", {}) or {}).get(
            "recognition", {}
        ) or {}
        # Automatic kernel routing is reserved for distributionally forced
        # problems. Homogeneous IVPs should use their condition-aware heat,
        # wave, transform, or series solvers; fundamental solutions remain
        # available through the explicit kernel APIs.
        if recognition.get("has_source", False):
            kscore = 95 if kernel_plan.method == "kernel_green_function" else 93
            add(
                kernel_plan.method,
                kscore,
                f"kernel plan: {kernel_plan.operator_family} on {kernel_plan.geometry_kind}",
                operator_family=kernel_plan.operator_family,
                geometry_kind=kernel_plan.geometry_kind,
                boundary_family=kernel_plan.boundary_family,
                kernel_plan=kernel_plan,
            )

    nonlinear_info = None
    recognitions = tuple(profile.details.get("recognitions", ()) or ())
    for rec in recognitions:
        if (
            rec.recognized
            and rec.solver_hint
            and rec.solver_hint not in {c.method for c in cands}
        ):
            # Family recognizers intentionally lack enough condition/domain
            # context to select these concrete solvers.  The contextual rules
            # below add them only when their actual applicability is known.
            if rec.solver_hint in {"heat_dirichlet_series", "wave_dalembert"}:
                continue
            bonus = (
                4
                if rec.family in {"generalized_clairaut", "scalar_conservation_law"}
                else 0
            )
            add(
                rec.solver_hint,
                70 + bonus,
                f"recognized structure: {rec.family}",
                **dict(rec.metadata),
            )
    cc_prof = profile.details.get("constant_coefficient_profile")
    if profile.order == 1:
        if profile.first_order_linear is not None:
            first_score = 76 if cc_prof is not None else 80
            add("first_order", first_score, "linear first-order form detected")
            if "profile" in ic_kinds:
                score = 96 if profile.first_order_linear.is_constant_coefficient else 74
                add("transport_ivp", score, "profile data supplied for first-order PDE")
        elif profile.characteristic_data is not None:
            add("first_order", 70, "first-order characteristic form detected")
        if profile.conservation_law is not None:
            cons_score = 60 if profile.first_order_linear is not None else 82
            add(
                "conservation_law",
                cons_score,
                "scalar conservation-law structure detected",
            )
            if fam == "inviscid_burgers" and "profile" in ic_kinds:
                add(
                    "burgers_implicit",
                    94,
                    "Burgers IVP supports implicit characteristic solution",
                )
        if profile.first_order_linear is None:
            try:
                from .first_order_nonlinear import analyze_first_order_nonlinear_pde

                nonlinear_info = analyze_first_order_nonlinear_pde(
                    eq_or_expr, dep_expr_or_func, profile.indep_vars
                )
            except Exception:
                nonlinear_info = None
            if nonlinear_info is not None and nonlinear_info.is_first_order:
                if nonlinear_info.is_quasilinear:
                    add(
                        "complete_integral",
                        78,
                        "quasilinear first-order nonlinear structure detected",
                    )
                    add(
                        "charpit",
                        84,
                        "Charpit method applicable to two-variable quasilinear first-order PDE",
                    )
                    if has_profile_data:
                        add(
                            "quasilinear_implicit",
                            98,
                            "quasilinear first-order IVP with profile data",
                        )
                else:
                    add(
                        "complete_integral",
                        72,
                        "first-order nonlinear PDE supports complete-integral search",
                    )
                    add(
                        "charpit",
                        80,
                        "first-order nonlinear PDE supports Charpit search",
                    )
                    if len(profile.indep_vars) >= 3:
                        add(
                            "jacobi",
                            76,
                            "higher-dimensional first-order PDE supports Jacobi complete-integral search",
                        )
                add(
                    "invariant_reduction_auto",
                    74,
                    "first-order nonlinear PDE supports invariant reduction",
                )
                add(
                    "first_order_nonlinear_auto",
                    86,
                    "first-order nonlinear auto solver available",
                )
    if cc_prof is not None:
        score = 92 if getattr(cc_prof, "is_constant_coefficient", True) else 70
        if fam == "heat_like" and has_profile_data:
            score = min(score, 84)
        add(
            "constant_coefficient_inverse_operator",
            score,
            "constant-coefficient operator profile detected",
        )

    if fam == "heat_like":
        if geom in {"whole_line", "full_line"} and has_profile_data:
            add("heat_whole_line", 90, "whole-line heat IVP via kernel")
            add(
                "fourier_heat",
                96 if prefer_transform else 88,
                "whole-line heat IVP via Fourier transform",
            )
            add("heat_laplace_transform", 79, "Laplace-in-time heat workflow available")
            add(
                "laplace_fourier_heat",
                82 if prefer_transform else 74,
                "combined Laplace/Fourier formal workflow available",
            )
        if geom == "half_line" and has_profile_data:
            add(
                "heat_half_line_transform",
                96 if prefer_transform else 92,
                "half-line heat equation with boundary data",
            )
        if (
            geom == "interval"
            and (
                ("dirichlet" in bc_kinds) or bc_type == "dirichlet_homogeneous_interval"
            )
            and has_profile_data
        ):
            add(
                "heat_dirichlet_series",
                98 if prefer_separation else 96,
                "interval Dirichlet heat problem",
                eigenbasis=getattr(sep_plan, "eigenbasis", None),
            )
        if (
            geom == "interval"
            and (("neumann" in bc_kinds) or bc_type == "neumann_homogeneous_interval")
            and has_profile_data
        ):
            add(
                "heat_neumann_series",
                97 if prefer_separation else 95,
                "interval Neumann heat problem",
                eigenbasis=getattr(sep_plan, "eigenbasis", None),
            )
        if geom == "interval" and {"robin"} <= bc_kinds and has_profile_data:
            add(
                "heat_robin_series",
                97 if prefer_separation else 94,
                "interval Robin heat problem",
                eigenbasis=getattr(sep_plan, "eigenbasis", None),
            )
        if geom == "rectangle" and {"dirichlet"} <= bc_kinds:
            add(
                "heat_dirichlet_series",
                84 if prefer_separation else 78,
                "rectangular Dirichlet heat problem",
            )
        if geom == "rectangle" and {"neumann"} <= bc_kinds:
            add(
                "heat_neumann_series",
                82 if prefer_separation else 76,
                "rectangular Neumann heat problem",
            )
        if geom == "rectangle" and {"robin"} <= bc_kinds:
            add(
                "heat_robin_series",
                81 if prefer_separation else 75,
                "rectangular Robin heat problem",
            )
    if fam == "wave_like":
        has_wave_ics = "profile" in ic_kinds and "velocity" in ic_kinds
        if isinstance(ics, dict):
            has_wave_ics = has_wave_ics or (
                "initial_displacement" in ics and "initial_velocity" in ics
            )
        if len(profile.indep_vars) == 2 and has_wave_ics:
            if geom == "interval" and (
                ("dirichlet" in bc_kinds) or bc_type == "dirichlet_homogeneous_interval"
            ):
                add(
                    "wave_dirichlet_series",
                    99 if prefer_separation else 98,
                    "finite-interval Dirichlet wave IVP",
                )
            else:
                add(
                    "wave_dalembert",
                    97,
                    "1D whole-line wave IVP with displacement and velocity data",
                )
    if fam in {
        "laplace_like",
        "helmholtz_like",
        "diffusion_reaction",
        "reaction_diffusion_like",
    }:
        add(
            "separation_of_variables",
            80 if prefer_separation else 72,
            "second-order PDE supports separation workflows",
        )
        if geom == "rectangle" and bc_kinds == {"dirichlet"}:
            add(
                "laplace_rectangle_dirichlet_series",
                94 if prefer_separation else 88,
                "rectangle Dirichlet Laplace problem",
                eigenbasis=getattr(sep_plan, "eigenbasis", None),
            )
    if profile.second_order_class is not None:
        add(
            "classification_only",
            20,
            f"classified as {profile.second_order_class.classification}",
        )
    else:
        add(
            "classification_only",
            5,
            "structural fallback when no executable solver applies",
        )
    if profile.principal_solved_form is not None:
        add(
            "symmetry_reduction",
            60 if profile.order > 1 else 52,
            "principal solved form available for symmetry reduction",
        )
        if prefer_symmetry:
            add("symmetry_reduction", 88, "symmetry preferred by caller")

    best_by_method: dict[str, PDESolverMethodCandidate] = {}
    for cand in cands:
        prev = best_by_method.get(cand.method)
        if prev is None or cand.score > prev.score:
            best_by_method[cand.method] = cand
    return profile, tuple(
        sorted(best_by_method.values(), key=lambda c: (-c.score, c.method))
    )


def plan_pde_solution_methods(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    domain=None,
    assumptions=True,
    prefer_transform=False,
    prefer_separation=False,
    prefer_symmetry=False,
):
    profile, ranked = rank_pde_solution_methods(
        eq_or_expr,
        dep_expr_or_func,
        indep_vars,
        ics=ics,
        bcs=bcs,
        domain=domain,
        assumptions=assumptions,
        prefer_transform=prefer_transform,
        prefer_separation=prefer_separation,
        prefer_symmetry=prefer_symmetry,
    )
    canonical = profile.details.get("canonical_representation")
    (
        _,
        _,
        _,
        cmodel,
        dgeom,
        condition_analysis,
        sep_plan,
        transform_plan,
        boundary_model,
        _kernel_plan,
    ) = _normalized_problem_geometry(
        profile.indep_vars,
        ics=ics,
        bcs=bcs,
        dep_expr=profile.dep_function,
        canonical=canonical,
        domain=domain,
    )
    return PDESolutionPlan(
        profile,
        tuple(ranked),
        {
            "ics": ics,
            "bcs": bcs,
            "domain": domain,
            "assumptions": assumptions,
            "condition_model": cmodel,
            "domain_geometry": dgeom,
            "condition_analysis": condition_analysis,
            "separation_plan": sep_plan,
            "transform_plan": transform_plan,
            "boundary_model": boundary_model,
        },
    )


def plan_pde_solution(problem):
    if hasattr(problem, "equation"):
        return plan_pde_solution_methods(
            problem.equation,
            problem.dep_function,
            problem.indep_vars,
            ics=getattr(problem, "ics", None),
            bcs=getattr(problem, "bcs", None),
            domain=getattr(problem, "domain", None),
            assumptions=getattr(problem, "assumptions", True),
        )
    raise TypeError("plan_pde_solution expects a PDEProblem-like object.")
