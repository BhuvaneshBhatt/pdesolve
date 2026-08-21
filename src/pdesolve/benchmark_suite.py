from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from .benchmark_cases import BenchmarkCase, build_benchmark_cases


@dataclass(frozen=True)
class BenchmarkOutcome:
    case: BenchmarkCase
    success: bool
    method: str | None
    message: str = ""
    exact_verified: bool = False
    contains_verified: bool = False
    method_hint_verified: bool = False


@dataclass(frozen=True)
class BenchmarkSuite:
    cases_by_family: dict[str, tuple[BenchmarkCase, ...]]

    @property
    def families(self):
        return tuple(self.cases_by_family.keys())

    def all_cases(self):
        for cases in self.cases_by_family.values():
            for case in cases:
                yield case

    @property
    def family_counts(self) -> dict[str, int]:
        return {fam: len(cases) for fam, cases in self.cases_by_family.items()}

    @property
    def total_cases(self) -> int:
        return sum(self.family_counts.values())

    @property
    def stress_cases(self) -> tuple[BenchmarkCase, ...]:
        return tuple(
            case for case in self.all_cases() if case.stress_level != "standard"
        )


def get_method_family_regression_cases():
    return build_benchmark_cases()


def build_benchmark_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        {k: tuple(v) for k, v in get_method_family_regression_cases().items()}
    )


def _normalize_solution_obj(sol: Any) -> Any:
    if hasattr(sol, "solutions"):
        sols = getattr(sol, "solutions")
        if isinstance(sols, (list, tuple)) and len(sols) == 1:
            return _normalize_solution_obj(sols[0])
    if hasattr(sol, "solution"):
        nested = getattr(sol, "solution")
        if nested is not sol:
            return _normalize_solution_obj(nested)
    if hasattr(sol, "rhs") and hasattr(sol, "lhs"):
        return sol
    return sol


def _compare_eqs(actual: Any, expected: Any) -> bool:
    try:
        if not (
            hasattr(actual, "lhs")
            and hasattr(actual, "rhs")
            and hasattr(expected, "lhs")
            and hasattr(expected, "rhs")
        ):
            return False
        same_lhs = sp.srepr(actual.lhs) == sp.srepr(expected.lhs)
        rhs_match = bool(sp.simplify(actual.rhs - expected.rhs) == 0)
        return same_lhs and rhs_match
    except Exception:
        return False


def _solution_matches_expected(actual: Any, expected: Any) -> bool:
    actual = _normalize_solution_obj(actual)
    expected = _normalize_solution_obj(expected)
    try:
        if isinstance(actual, dict) and isinstance(expected, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(
                _solution_matches_expected(actual[k], expected[k]) for k in actual
            )
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            return len(actual) == len(expected) and all(
                _solution_matches_expected(a, e) for a, e in zip(actual, expected)
            )
        if (
            hasattr(actual, "lhs")
            and hasattr(actual, "rhs")
            and hasattr(expected, "lhs")
            and hasattr(expected, "rhs")
        ):
            return _compare_eqs(actual, expected)
        return bool(sp.simplify(actual - expected) == 0)
    except Exception:
        try:
            return sp.srepr(actual) == sp.srepr(expected)
        except Exception:
            return False


def _contains_tokens(actual: Any, tokens: tuple[str, ...]) -> bool:
    if not tokens:
        return True
    text = str(actual)
    return any(tok in text for tok in tokens)


def _method_matches_hints(method: str | None, hints: tuple[str, ...]) -> bool:
    if not hints:
        return True
    if method is None:
        return False
    lower = method.lower()
    return any(h.lower() in lower for h in hints)


def run_benchmark_case(case: BenchmarkCase) -> BenchmarkOutcome:
    from .dispatcher import pdesolve

    try:
        res = pdesolve(
            case.equation, case.dependent, case.variables, ics=case.ics, bcs=case.bcs
        )
        if res is None:
            return BenchmarkOutcome(
                case=case, success=False, method=None, message="solver returned None"
            )
        method = str(getattr(res, "method", None)) if res is not None else None
        solution = getattr(res, "solution", res)
        exact_verified = (
            True
            if case.expected_solution is None
            else _solution_matches_expected(solution, case.expected_solution)
        )
        contains_verified = _contains_tokens(solution, case.solution_fragments)
        method_hint_verified = _method_matches_hints(method, case.expected_method_hints)
        success = (
            (res is not None)
            and exact_verified
            and contains_verified
            and method_hint_verified
        )
        message_parts = []
        if not exact_verified:
            message_parts.append("exact solution mismatch")
        if not contains_verified:
            message_parts.append("solution text missing expected tokens")
        if not method_hint_verified:
            message_parts.append("method did not match expected hints")
        return BenchmarkOutcome(
            case=case,
            success=success,
            method=method,
            message="; ".join(message_parts),
            exact_verified=exact_verified,
            contains_verified=contains_verified,
            method_hint_verified=method_hint_verified,
        )
    except Exception as exc:
        return BenchmarkOutcome(case=case, success=False, method=None, message=str(exc))


def run_benchmark_suite() -> tuple[BenchmarkOutcome, ...]:
    suite = build_benchmark_suite()
    return tuple(run_benchmark_case(case) for case in suite.all_cases())


__all__ = [
    "BenchmarkCase",
    "BenchmarkOutcome",
    "BenchmarkSuite",
    "get_method_family_regression_cases",
    "build_benchmark_suite",
    "run_benchmark_case",
    "run_benchmark_suite",
]
