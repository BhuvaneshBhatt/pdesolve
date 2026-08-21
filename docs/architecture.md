# Architecture

## End-to-end pipeline

```text
User equations / conditions
        ↓
build_pde_problem
        ↓
PDEProblem + canonical representation + PDEProblemProfile
        ↓
recognition + structured condition/domain/boundary models
        ↓
rank_pde_solution_methods / plan_canonical_problem
        ↓
PDESolutionPlan.steps
        ↓
execute_planned_solver / solve_with_canonical_problem
        ↓
method-specific solver or framework delegate
        ↓
BasePDEResult subclass + metadata + verification + trace
```

## Core orchestration modules

- `problem.py` — constructs `PDEProblem`, preserves normalized/canonical metadata.
- `canonical.py`, `recognition.py`, `classification.py` — canonical representation, recognition, family refinement, ranking, and plans.
- `conditions.py`, `domains.py`, `boundary_model.py`, `condition_analysis.py` — structured problem data and diagnostics.
- `dispatcher.py` — input parsing and public `pdesolve(...)` entry point.
- `solve_pipeline.py` — method ordering, execution attempts, result verification, and trace finalization.
- `solver_execution.py` — canonical method registry and result standardization.
- `recognizers/coordinator.py`, `planners/coordinator.py`, `solvers/coordinator.py` — thin coordinator APIs over the canonical layers.

## Method subsystems

- `classical_first_order.py`, `first_order_linear.py`, `first_order_nonlinear.py`, `first_order_framework.py`, `first_order_geometry.py`
- `complete_integral_helpers.py`
- `conservation_laws.py`
- `constant_coeff.py`, `operator_symbol.py`
- `classical_evolution.py`, `classical_classification.py`, `separation_framework.py`, `ivp_bvp.py`
- `transform_framework.py`, `transforms.py`, `unified_transform.py`
- `hyperbolic_system.py`
- `kernels.py`, `green_subsystem.py`
- symmetry/reduction modules including `symmetry.py`, `reduction.py`, `lie.py`, `frobenius.py`, and differential-invariant helpers

## Why frameworks exist

`separation_framework` and `structured_transform` are planning/execution coordinators. They use normalized conditions and geometry to select a concrete calculation. They should not obscure the concrete method identity in the final user-facing result when a more specific name is available.

## Caching and performance

Preprocessing/classification includes memoized paths for repeated symbolic work. Expensive symbolic verification is intentionally avoided where the result type makes a generic residual simplification unsuitable.
