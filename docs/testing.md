# Testing and benchmarks

## Running tests

```bash
python -m pytest -q
```

For rapid failure isolation:

```bash
python -m pytest -x -q
```

## What the regression suite covers

The current suite contains tests for:

- canonical problem construction and coordinator layers,
- first-order linear and nonlinear methods,
- Charpit/Jacobi/complete-integral/Cauchy workflows,
- scalar conservation laws, shocks, rarefactions, and weak verification,
- constant-coefficient operator methods and resonance handling,
- heat/wave series and transform methods,
- separation and structured transform planning,
- unified transform,
- hyperbolic systems,
- symmetry/invariant/Frobenius reduction infrastructure,
- condition/domain/boundary consistency,
- fundamental solutions and Green functions,
- standardized result/trace behavior,
- public API contracts,
- benchmark suites,
- a 52-example reference PDE matrix.

## Reference matrix

`tests/test_reference_pde_examples.py` contains 52 heterogeneous examples spanning first-order equations, nonlinear equations, wave/heat problems, multidimensional equations, systems, Laplace/Helmholtz-style BVPs, finance/quantum-style forms, and robustness cases. Its primary assertion is that the dispatcher returns a meaningful method/result; more targeted files test mathematical details for individual solver families.

A passing heterogeneous reference example should therefore be read as **routing/execution evidence**, not as proof that every member of the corresponding PDE family is solved.

## Benchmarks

`pdesolve.benchmark_suite` exposes:

- `build_benchmark_suite()`
- `run_benchmark_case(...)`
- `run_benchmark_suite()`
- `get_method_family_regression_cases()`

Benchmark cases include expected methods, exact-output metadata where appropriate, reduced-equation regressions, and stress tags. Exact comparisons should use mathematically equivalent canonical expectations rather than brittle formatting identities.

## Verification tests

Tests distinguish ordinary explicit residual verification from weak/admissibility, kernel, implicit, or formal-result semantics. This is intentional: verification policy is part of the result-type contract.

## worked examples executable-example policy

Worked examples are selected from regression-tested solver paths. When a focused direct API has more specialized support than automatic routing, the documentation says so explicitly rather than presenting a direct capability as an `auto` capability. Code examples should be kept synchronized with the corresponding tests when method names or result semantics change.


## PDESolve documentation notebook regression policy

The maintained tutorial curriculum lives in `notebooks/tutorials/`. Every family notebook contains executable mathematical assertions, and `tools/execute_tutorial_notebooks.py` can execute the complete curriculum. The normal pytest suite performs lightweight structural checks on the committed notebooks rather than launching eleven Jupyter kernels on every test run.

Before publishing documentation changes that affect solver examples, run:

```bash
python tools/execute_tutorial_notebooks.py --check
python -m pytest -q tests/test_tutorial_notebooks.py
```

Notebook execution is an integration/documentation test: it supplements, rather than replaces, focused unit tests for solver mathematics.

## Robustness and documentation contracts

Regression tests cover cache isolation and clearing, bounded preprocessing state, variable-order-independent initial-condition semantics, unspecified-domain behavior, obligation-aware verification, and the distinction between expected solver inapplicability and unexpected programming errors.

Documentation tests also enforce the canonical result API: maintained Markdown examples and tutorial notebooks must use `result.metadata` rather than `details`. This prevents examples from drifting back toward a second result-metadata convention.

The formal wave Laplace-sine regression additionally checks that modal initial-data objects are represented as SymPy functions (`F_n(n)` and `G_n(n)`) rather than callable scalar symbols.
