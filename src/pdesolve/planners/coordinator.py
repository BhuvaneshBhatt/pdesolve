from __future__ import annotations

from ..classification import plan_pde_solution_methods


def plan_canonical_problem(problem, **preferences):
    if getattr(problem, "profile", None) is None:
        # system problems route directly today
        class _Plan:
            def __init__(self):
                self.profile = None
                self.steps = ()
                self.details = dict(getattr(problem, "details", {}) or {})

        return _Plan()
    return plan_pde_solution_methods(
        problem.equation,
        problem.dep_function,
        problem.indep_vars,
        ics=problem.ics,
        bcs=problem.bcs,
        domain=getattr(problem, "domain", None),
        assumptions=problem.assumptions,
        prefer_transform=preferences.get("prefer_transform", False),
        prefer_separation=preferences.get("prefer_separation", False),
        prefer_symmetry=preferences.get("prefer_symmetry", False),
    )
