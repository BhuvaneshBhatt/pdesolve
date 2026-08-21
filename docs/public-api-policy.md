# Public API policy

PDESolve exposes its supported package-level API through `pdesolve.__all__`. Public names should represent stable mathematical concepts, solver entry points, result types, or inspection utilities rather than internal implementation stages.

API changes should be intentional and accompanied by updates to tests, documentation, and `test_reports/public-api-audit.json`. The audit tool can verify that the documented export set matches the package:

```bash
python tools/public_api_audit.py --check
```

Internal helpers should remain module-private unless users need them to construct, inspect, or extend solver behavior.
