# Methods and results

This former combined page has been split for documentation:

- [Capability matrix](capability-matrix.md)
- [Methods and solver inventory](method-inventory.md)
- [Results and verification](results-verification.md)
- [Planning and diagnostics](planning-diagnostics.md)
- [Limitations](limitations.md)

The split is intentional: mathematical capability, executable method keys, planner labels, and result semantics are related but not identical concepts.

## Cross-method contracts

Regardless of the selected mathematical method, standardized results follow the same contracts:

- `metadata` is the canonical read-only diagnostic store;
- verification is obligation-aware and may be `True`, `False`, or `None`;
- unspecified domain information remains unspecified rather than being interpreted as a whole-line problem;
- expected method inapplicability is distinct from an unexpected implementation exception.

`post_reduction_auto` is an execution route for an equation that has already been reduced. It is not an ordinary top-level candidate for an unreduced PDE.
