# PDESolve documentation

PDESolve is a symbolic PDE toolbox built on SymPy. PDESolve documentation builds on the 2.0 inventory and 2.1 worked examples and is organized around the package that actually exists today: the exported API, canonical recognizers/planner, executable method registry, direct solver APIs, and regression tests.

## Recommended path

1. [Getting started](getting-started.md) — solve a PDE and inspect the result.
2. [Capability matrix](capability-matrix.md) — what is supported, how strongly, and through which interface.
3. [Inputs, conditions, and domains](inputs-conditions-domains.md) — how problems are represented.
4. [Methods and solver inventory](method-inventory.md) — automatic methods, direct APIs, and formal methods.
5. [Results and verification](results-verification.md) — result classes, method identity, trace, and verification semantics.
6. [Planning and diagnostics](planning-diagnostics.md) — inspect recognition, ranking, condition analysis, and execution choices.
7. [Public API inventory](public-api-inventory.md) — all top-level exports and signatures.
8. [Architecture](architecture.md) — canonicalization → recognition → planning → execution.
9. [Limitations and support boundaries](limitations.md) — partial, formal, and unsupported cases.
10. [Developer guide](developer-guide.md) — extending recognizers, planners, solvers, and tests.
11. [Executable tutorial notebooks](tutorial-notebooks.md) — derivations, planner introspection, verification, plots, and exercises.
12. [Testing and benchmarks](testing.md) — regression strategy and benchmark suite.

## Support labels used in these docs

- **Verified path**: an implementation path exists and dedicated regression tests exercise it.
- **Supported path**: implementation and routing exist, but coverage is narrower or concentrated in broader integration tests.
- **Direct API**: callable support exists but is not necessarily selected by `pdesolve(..., method="auto")`.
- **Formal/partial**: the package can construct a representation, reduction, transform, or candidate answer, but does not promise a fully verified closed form.
- **Recognition only**: PDESolve can classify or canonicalize the problem without necessarily solving it.

These labels describe the current implementation and tests, not mathematical solvability in general.

## Learn by example

The [worked mathematical examples](examples/index.md) trace representative problems from recognition and planning through result interpretation, verification, and limitations. PDESolve documentation continues those examples as [fully executable tutorial notebooks](tutorial-notebooks.md).

## Symbolic infrastructure

See [Symbolic infrastructure](symbolic-infrastructure.md) for verification strategies, general separation, Sturm–Liouville support, transform postprocessing, and Lie point-symmetry analysis.
