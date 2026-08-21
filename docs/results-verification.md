# Results, method identity, and verification

## Standard result model

Canonical execution standardizes solver outputs into subclasses of `BasePDEResult`. The main public result classes include:

- `ClosedFormPDEResult`
- `ImplicitPDEResult`
- `SeriesPDEResult`
- `TransformPDEResult`
- `WeakSolutionResult`
- `EigenfunctionExpansionResult`
- `SystemPDEResult`
- `NumericalFallbackResult`
- `UnsolvedButReducedResult`
- `FundamentalSolutionResult`
- `GreenFunctionResult`
- `PDESolutionRecord`

Common fields include `method`, `solution`, `classification`, `assumptions`, `verification`, `metadata`, `reduced_problem`, and `warnings`.

## Concrete method versus coordinator method

A planner may choose `separation_framework` or `structured_transform`, which can delegate to a concrete heat/Fourier/series solver. Public results should report the concrete method when that identity is meaningful; coordinator/planning information belongs in metadata/trace. This distinction keeps result interpretation precise.

## Verification is typed, not binary

Verification can include different checks depending on the result:

- explicit PDE residual substitution/simplification,
- initial/boundary condition checks,
- complete-integral branch verification,
- weak/admissibility checks for conservation laws,
- kernel boundary/source metadata checks,
- method-specific structural checks,
- inconclusive/unverified status when a generic symbolic check is inappropriate.

An empty or inconclusive verification record is not the same as “incorrect”. It means the package has not established the relevant condition by its available checker.

## Why implicit/weak solutions are special

Implicit characteristic relations can be self-referential after substitution, and weak solutions are not validated by ordinary pointwise residual simplification alone. PDESolve therefore avoids forcing every result through the same expensive `simplify()`-style path. This both prevents pathological runtimes and better matches the mathematical semantics of the result class.

## Execution trace

`extract_solution_trace(...)` and `summarize_solution_record(...)` expose trace/provenance information attached during dispatch. Use them when diagnosing why a method was selected or how a standardized result was produced.

## Verification obligations

Overall verification is obligation-aware. The PDE residual is required whenever an equation is supplied; initial-data checks are required when initial conditions are supplied; and boundary checks are required when boundary conditions are supplied.

The aggregate state is three-valued:

- `True` only when every required obligation was checked and passed;
- `False` when at least one required obligation was checked and failed;
- `None` when no required obligation failed but at least one required check is unavailable or inconclusive.

Consequently, a successful PDE residual alone cannot make a result verified when the problem also supplied initial or boundary data that were not checked.

## Result metadata

`PDESolutionRecord.metadata` is the canonical store for solver diagnostics, execution traces, normalization reports, and method-specific structured data. The mapping is read-only on returned results so a caller cannot accidentally mutate a result after construction.

Examples and new code should use `result.metadata`. The `details` attribute is only another read-only view of the same mapping; it is not a second independently maintained store and should not be used as the documentation-facing API.

## Solver exceptions

The public exception hierarchy is rooted at `PDESolveError` and includes `PDEInputError`, `PDEMethodNotApplicable`, `PDETransformationError`, and `PDEVerificationError`.

`PDEMethodNotApplicable` means that a solver was considered correctly but its mathematical preconditions were not met. `PDETransformationError` represents an expected failure while constructing a supported transformation. Auto-dispatch may catch those expected failures and continue to another planned method. Invalid input is reported with `PDEInputError`, while unexpected programming errors such as accidental `AttributeError` or `TypeError` are allowed to propagate rather than being mislabeled as solver inapplicability.
