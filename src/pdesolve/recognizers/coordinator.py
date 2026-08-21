from __future__ import annotations

from ..recognition import build_canonical_representation, recognize_pde_structure


def recognize_canonical_problem(problem):
    canonical = getattr(problem, "canonical_representation", None)
    if canonical is None and getattr(problem, "profile", None) is not None:
        canonical = build_canonical_representation(
            problem.profile,
            ics=problem.ics,
            bcs=problem.bcs,
            normalized_data=getattr(problem, "normalized_data", None),
            dep_expr=problem.dep_function,
        )
    if canonical is None:
        return ()
    recs = (
        tuple((getattr(problem, "profile", None) or {}).details.get("recognitions", ()))
        if getattr(problem, "profile", None) is not None
        else ()
    )
    if not recs and getattr(problem, "profile", None) is not None:
        recs = recognize_pde_structure(problem.profile, canonical=canonical)
    return recs
