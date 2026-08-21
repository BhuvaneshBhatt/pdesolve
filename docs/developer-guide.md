# Developer guide

## Architectural contract

New methods should respect the canonical pipeline:

`PDEProblem → canonical representation/profile → recognition → planning → solver execution → standardized result → verification/trace`

Avoid adding a second independent dispatcher unless a focused direct API genuinely requires it.

## Adding a recognizer

A recognizer should:

1. normalize rather than mutate caller input;
2. return explicit structured metadata;
3. separate “recognized” from “solved”;
4. provide a solver hint only when an execution path exists;
5. have positive and negative regression tests.

If recognition depends on conditions or geometry, prefer a structured plan object (`ConditionModel`, `DomainGeometry`, `BoundaryModel`) over reparsing raw equalities.

## Adding a planner method

Before adding a method name to ranking logic, verify that it is either:

- present in the canonical execution registry, or
- deliberately translated/delegated before canonical execution.

Add a test that takes the ranked candidate through `solve_with_canonical_problem(...)` or the public dispatcher. This prevents “planner-only” method labels from becoming dead routes.

## Adding a solver

A canonical solver handler should return either a standard result object or a method-specific object that `_standardize_result(...)` can convert without losing method identity, verification data, warnings, and metadata.

Prefer concrete method names in the final result. Keep coordinator/framework names in metadata when the framework delegated to a more specific solver.

## Verification hooks

Do not apply one universal simplifier to all result types. Choose verification according to semantics:

- pointwise residual for explicit classical solutions,
- branch/implicit checks for complete integrals,
- weak/admissibility checks for conservation laws,
- source/boundary identities for kernels,
- structural checks for transforms/reductions.

Add time/complexity guards around expensive symbolic verification paths.


## Conditions, domains, and evolution variables

Do not infer the evolution variable from its position in `indep_vars`. Initial-condition analysis derives it from the initial surface, so equivalent problems remain equivalent when variables are supplied as `(x, t)` or `(t, x)`.

Likewise, absence of an explicit or inferable spatial domain means the geometry is `unspecified`. Do not silently promote an unknown one-dimensional domain to the full real line; whole-line, half-line, and finite-interval methods should be selected only when the geometry is established.

## Expected solver failures

Use the public exception hierarchy to distinguish mathematical inapplicability from implementation defects. A solver whose preconditions are not met should raise `PDEMethodNotApplicable`; a supported transformation that cannot be completed should raise `PDETransformationError`. Invalid caller data should raise `PDEInputError`.

Do not broadly catch `Exception` at planner/dispatcher boundaries. Unexpected exceptions should remain visible so programming errors are not converted into ordinary method rejection.

## Result metadata

Store method diagnostics and structured auxiliary information in the result `metadata` mapping. Returned metadata is read-only. Do not create a parallel mutable result-details store, and use `result.metadata` in new tests, examples, and documentation.

## Preprocessing cache

Preprocessing uses a bounded cache. Cached state must not expose mutable dictionaries or other caller-mutable objects that can contaminate later solves. Cache-clearing utilities used by benchmarks/tests must clear preprocessing state as well as the specialized symbolic caches.

## Tests expected for new capabilities

A substantial new solver should normally include:

- recognizer/canonicalization test,
- planner-ranking test,
- direct solver test,
- public `pdesolve(...)` integration test,
- result-class/method-identity assertion,
- verification assertion appropriate to the method,
- at least one negative or unsupported case,
- benchmark/reference case when useful.

## Documentation expected for new capabilities

Update at least:

- `capability-matrix.md`,
- `method-inventory.md`,
- `public-api-inventory.md` if exports change,
- `limitations.md` if support remains partial/formal.

Run `python tools/generate_doc_inventory.py --output docs/generated-inventory.json` to regenerate the machine-readable export/method/test-reference inventory. The human-facing API table is derived from the same package surface; do not hand-maintain an export count without checking `pdesolve.__all__`.
## Repository layout

PDESolve uses a `src` layout. Importable runtime code lives in `src/pdesolve/`; tests, documentation, notebooks, examples, and maintenance tools remain at repository level. Install the package before running tests so imports exercise the package configuration rather than the checkout directory itself.

```bash
python -m pip install -e .
pytest
```

This arrangement helps catch missing package data and package-discovery mistakes before publication.

