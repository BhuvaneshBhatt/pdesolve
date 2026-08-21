# Planning, recognition, and diagnostics

## Pipeline inspection

```python
problem = pds.build_pde_problem(data, dep_expr, indep_vars)
recs = pds.recognize_canonical_problem(problem)
plan = pds.plan_canonical_problem(problem)
```

A `PDESolutionPlan` contains ranked `PDESolverMethodCandidate` steps. Scores encode preference among applicable methods, not mathematical certainty.

## Major planning signals

The planner currently considers, among other data:

- canonical PDE family and differential order,
- first-order linear/characteristic structure,
- conservation-law structure,
- constant-coefficient operator profile,
- initial-condition kinds (profile, velocity, etc.),
- inferred geometry (full line, half line, interval, rectangle, disk/annulus),
- boundary family (Dirichlet, Neumann, Robin),
- separation and transform plans,
- source terms suitable for kernel routing,
- explicit user preferences for transforms, separation, or symmetry.

## Preference flags

`prefer_transform=True`, `prefer_separation=True`, and `prefer_symmetry=True` adjust ranking. They do not create mathematical applicability where recognition failed.

## Kernel routing is conservative

Automatic kernel routing is reserved primarily for distributionally forced problems recognized as fundamental/Green-function requests. Ordinary homogeneous heat or wave IVPs should route to condition-aware IVP/transform/series methods instead. Direct kernel APIs remain available at any time.

## Classification-only and reduced results

If classification or reduction succeeds but no downstream solver completes the problem, PDESolve can return `UnsolvedButReducedResult`. This is useful information: the package has transformed or classified the problem without claiming a final solution.

## Debugging a surprising route

1. Inspect `problem.profile` and canonical metadata.
2. Inspect `ConditionModel`, `DomainGeometry`, and `BoundaryModel` in problem/plan details.
3. Print every `plan.steps` entry with its score and reasons.
4. Compare the selected method with the canonical execution registry in [method-inventory.md](method-inventory.md).
5. Inspect `result.metadata`, `result.warnings`, and the execution trace.
6. If necessary, call a focused direct API to separate routing behavior from solver behavior.
