from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import sympy as sp

from .results import (
    PDEVerificationSummary,
    BasePDEResult,
    ImplicitPDEResult,
    WeakSolutionResult,
    KernelRepresentationResult,
    SeriesPDEResult,
    TransformPDEResult,
)
from .verify import verify_solution_with_conditions, verify_kernel_representation


class VerificationStrategy(Protocol):
    name: str

    def verify(
        self,
        equation,
        result,
        dep_function,
        indep_vars,
        *,
        ics=None,
        bcs=None,
        assumptions=True,
    ) -> PDEVerificationSummary: ...


@dataclass(frozen=True)
class ClassicalResidualVerifier:
    name: str = "classical_residual"

    def verify(
        self,
        equation,
        result,
        dep_function,
        indep_vars,
        *,
        ics=None,
        bcs=None,
        assumptions=True,
    ):
        solution = getattr(result, "solution", result)
        return verify_solution_with_conditions(
            equation,
            solution,
            dep_function,
            indep_vars,
            ics=ics,
            bcs=bcs,
            assumptions=assumptions,
        )


@dataclass(frozen=True)
class ImplicitSolutionVerifier:
    name: str = "implicit_relation"

    def verify(
        self,
        equation,
        result,
        dep_function,
        indep_vars,
        *,
        ics=None,
        bcs=None,
        assumptions=True,
    ):
        # Generic implicit differentiation is expensive and method dependent.  Preserve any
        # method-specific verification supplied by the solver; otherwise report inconclusive.
        ver = getattr(result, "verification", None)
        if isinstance(ver, dict) and ver.get("verified") is not None:
            return PDEVerificationSummary(
                bool(ver["verified"]),
                "verified" if ver["verified"] else "failed",
                mode=self.name,
                message=ver.get("message", "method-specific implicit verification"),
            )
        return PDEVerificationSummary(
            None,
            "unverified",
            mode=self.name,
            message="Implicit representation requires method-specific differentiation/branch assumptions.",
        )


@dataclass(frozen=True)
class WeakSolutionVerifier:
    name: str = "weak_solution"

    def verify(
        self,
        equation,
        result,
        dep_function,
        indep_vars,
        *,
        ics=None,
        bcs=None,
        assumptions=True,
    ):
        try:
            from .conservation_laws import verify_weak_conservation_law_solution

            structured = (
                getattr(result, "metadata", {}).get("raw_result")
                if isinstance(result, BasePDEResult)
                else result
            )
            if structured is None:
                structured = result
            return verify_weak_conservation_law_solution(
                structured, dep_function, indep_vars
            )
        except Exception as exc:
            return PDEVerificationSummary(
                None,
                "unverified",
                mode=self.name,
                message=f"weak verification unavailable: {exc}",
            )


@dataclass(frozen=True)
class KernelVerifier:
    name: str = "kernel"

    def verify(
        self,
        equation,
        result,
        dep_function,
        indep_vars,
        *,
        ics=None,
        bcs=None,
        assumptions=True,
    ):
        kernel = getattr(result, "kernel", None) or getattr(result, "solution", result)
        try:
            info = verify_kernel_representation(
                equation,
                kernel,
                dep_function,
                indep_vars,
                bcs=bcs,
                operator_family=getattr(result, "operator_family", None),
                boundary_family=getattr(result, "boundary_type", None),
            )
            verified = info.get("verified")
            return PDEVerificationSummary(
                verified,
                "verified"
                if verified is True
                else "failed"
                if verified is False
                else "unverified",
                mode=self.name,
                message="kernel/source and boundary checks",
                boundary_residuals=tuple(info.get("boundary_residuals", ()) or ()),
            )
        except Exception as exc:
            return PDEVerificationSummary(
                None, "unverified", mode=self.name, message=str(exc)
            )


@dataclass(frozen=True)
class SeriesVerifier:
    name: str = "series"

    def verify(
        self,
        equation,
        result,
        dep_function,
        indep_vars,
        *,
        ics=None,
        bcs=None,
        assumptions=True,
    ):
        # Finite/truncated series can satisfy the PDE and BCs exactly while approximating data.
        base = verify_solution_with_conditions(
            equation,
            getattr(result, "solution", result),
            dep_function,
            indep_vars,
            ics=ics,
            bcs=bcs,
            assumptions=assumptions,
        )
        return PDEVerificationSummary(
            base.verified,
            base.status,
            base.pde_verified,
            base.initial_verified,
            base.boundary_verified,
            base.pde_residual,
            base.initial_residuals,
            base.boundary_residuals,
            self.name,
            base.message
            or "series verification distinguishes PDE/BC residuals from initial-data truncation",
        )


@dataclass(frozen=True)
class TransformVerifier:
    name: str = "transform_structural"

    def verify(
        self,
        equation,
        result,
        dep_function,
        indep_vars,
        *,
        ics=None,
        bcs=None,
        assumptions=True,
    ):
        sol = getattr(result, "solution", result)
        has_transform = isinstance(sol, sp.Basic) and bool(
            sol.has(sp.Integral, sp.InverseLaplaceTransform, sp.InverseFourierTransform)
        )
        return PDEVerificationSummary(
            None,
            "unverified",
            mode=self.name,
            message="Formal transform representation structurally recognized."
            if has_transform
            else "Transform result requires method-specific inversion/contour assumptions.",
        )


def select_verification_strategy(result: Any) -> VerificationStrategy:
    method = str(getattr(result, "method", "")).lower()
    if isinstance(result, WeakSolutionResult) or any(
        k in method for k in ("shock", "rarefaction", "conservation")
    ):
        return WeakSolutionVerifier()
    if isinstance(result, ImplicitPDEResult) or "implicit" in method:
        return ImplicitSolutionVerifier()
    if isinstance(result, KernelRepresentationResult) or any(
        k in method for k in ("kernel", "green", "fundamental_solution")
    ):
        return KernelVerifier()
    if (
        isinstance(result, SeriesPDEResult)
        or "series" in method
        or "separation" in method
    ):
        return SeriesVerifier()
    if isinstance(result, TransformPDEResult) or any(
        k in method for k in ("transform", "fourier", "laplace")
    ):
        return TransformVerifier()
    return ClassicalResidualVerifier()


def verify_result(
    equation, result, dep_function, indep_vars, *, ics=None, bcs=None, assumptions=True
) -> PDEVerificationSummary:
    """Verify a solver result with a representation-aware strategy."""
    strategy = select_verification_strategy(result)
    return strategy.verify(
        equation,
        result,
        dep_function,
        indep_vars,
        ics=ics,
        bcs=bcs,
        assumptions=assumptions,
    )


__all__ = [
    "VerificationStrategy",
    "ClassicalResidualVerifier",
    "ImplicitSolutionVerifier",
    "WeakSolutionVerifier",
    "KernelVerifier",
    "SeriesVerifier",
    "TransformVerifier",
    "select_verification_strategy",
    "verify_result",
]
