from __future__ import annotations

from ..errors import PDEError, PDEMethodNotApplicable
from ..solver_execution import solve_with_canonical_problem


def execute_planned_solver(problem, method, **kwargs):
    try:
        return solve_with_canonical_problem(problem, method, **kwargs)
    except PDEError:
        raise
    except (NotImplementedError, ValueError) as exc:
        raise PDEMethodNotApplicable(str(exc)) from exc
