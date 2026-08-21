from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pdesolve.solver_execution as solver_execution

ROOT = Path(__file__).resolve().parents[1]


def _load_hardening_module():
    path = ROOT / "tests" / "test_canonical_method_hardening.py"
    spec = importlib.util.spec_from_file_location(
        "pdesolve_method_hardening_tests", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_report() -> dict:
    module = _load_hardening_module()
    methods = tuple(solver_execution._METHOD_REGISTRY)
    specs = module.METHOD_COVERAGE_SPECS
    obligations = (
        "registered_and_documented",
        "dispatches_to_registered_handler",
        "representative_mathematical_execution",
        "declared_family_contract",
    )
    records = []
    for method in methods:
        spec = specs.get(method)
        records.append(
            {
                "method": method,
                "family": getattr(spec, "family", None),
                "test_count": len(obligations),
                "tests": [
                    f"tests/test_canonical_method_hardening.py::test_every_canonical_method_is_registered_and_documented[{method}]",
                    f"tests/test_canonical_method_hardening.py::test_every_canonical_method_dispatches_to_its_registered_handler[{method}]",
                    f"tests/test_canonical_method_hardening.py::test_every_canonical_method_has_representative_mathematical_execution[{method}]",
                    f"tests/test_canonical_method_hardening.py::test_every_canonical_method_has_declared_family_contract[{method}]",
                ],
            }
        )
    return {
        "canonical_method_count": len(methods),
        "minimum_tests_per_method": len(obligations),
        "registry_matches_coverage_specs": set(methods) == set(specs),
        "methods": records,
    }


def check_report(report: dict) -> list[str]:
    errors = []
    if not report["registry_matches_coverage_specs"]:
        errors.append("Canonical registry and coverage specs differ.")
    minimum = report["minimum_tests_per_method"]
    for record in report["methods"]:
        if record["test_count"] < minimum:
            errors.append(
                f"{record['method']} has only {record['test_count']} coverage tests."
            )
        if not record["family"]:
            errors.append(f"{record['method']} has no declared family.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or validate canonical-method test coverage."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "test_reports" / "method-coverage.json"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    report = build_report()
    errors = check_report(report)
    if args.check:
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(
            f"Method coverage OK: {report['canonical_method_count']} methods, "
            f">={report['minimum_tests_per_method']} explicit tests per method."
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
