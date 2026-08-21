# API guide

For a literal inventory of every package-level export and signature, see [Public API inventory](public-api-inventory.md).

## Primary user API

- `pdesolve(...)` — solve/route a PDE problem.
- `build_pde_problem(...)` — normalize and inspect a problem without committing to a solver.
- `recognize_canonical_problem(...)` — inspect structural recognition.
- `plan_canonical_problem(...)` — obtain ranked method candidates.
- `execute_planned_solver(...)` / `solve_with_canonical_problem(...)` — execute a canonical method explicitly.

## Focused solving APIs

Use focused functions when method control matters: first-order linear/nonlinear solvers, complete-integral/Cauchy solvers, conservation-law solvers, unified transform, hyperbolic systems, and kernel/Green-function APIs are all exported at package level.

## Structured data APIs

`ConditionModel`, `DomainGeometry`, `BoundaryModel`, `SeparableGeometryPlan`, `TransformMethodPlan`, `CanonicalFirstOrderPDE`, `CanonicalLinearSystemPDE`, and the result classes make the planner/executor state inspectable rather than opaque.



## Problem geometry and conditions

`build_pde_problem(...)` and the canonical planning pipeline preserve an unspecified domain as unspecified. They do not assume that a one-dimensional spatial variable ranges over the full real line merely because no domain was supplied. Specify `domain=` when whole-line, half-line, interval, or product geometry is part of the mathematical problem.

Initial-condition semantics are based on the initial surface rather than the positional order of independent variables, so callers do not need to place the evolution variable last.

## Results and exceptions

Use `result.metadata` for method-specific diagnostics and structured auxiliary data. Returned metadata is read-only.

The solver exception hierarchy is rooted at `PDESolveError`:

- `PDEInputError` — invalid problem or caller input;
- `PDEMethodNotApplicable` — a solver's mathematical preconditions are not satisfied;
- `PDETransformationError` — an expected failure while constructing a solver transformation;
- `PDEVerificationError` — a verification operation itself cannot be completed as required.

Automatic routing may continue after expected method-inapplicability or transformation failures. Unexpected programming exceptions are not converted into ordinary solver rejection.
