from __future__ import annotations

from typing import Any

import sympy as sp

from .results import (
    PDEExecutionTrace,
    PDEVerificationSummary,
    PDESolverTraceStep,
    PDESolutionRecord,
    BasePDEResult,
)
from .result_verification import verify_result
from .normalization import NormalizationPolicy, normalize_solution


def as_verification_summary(
    ver: Any,
    *,
    require_pde: bool = False,
    require_initial: bool = False,
    require_boundary: bool = False,
) -> PDEVerificationSummary:
    if isinstance(ver, PDEVerificationSummary):
        return ver
    if isinstance(ver, dict):
        verified = ver.get("verified")
        pde_res = ver.get("pde_residual")
        init_res = tuple(ver.get("initial_residuals", ()) or ())
        bc_res = tuple(ver.get("boundary_residuals", ()) or ())
        pde_ok = ver.get("pde_verified")
        init_ok = ver.get("initial_verified")
        bc_ok = ver.get("boundary_verified")
        if pde_ok is None and pde_res is not None:
            try:
                pde_ok = sp.simplify(pde_res) == 0
            except Exception:
                pde_ok = None
        if init_ok is None and init_res:
            try:
                init_ok = all(sp.simplify(item) == 0 for item in init_res)
            except Exception:
                init_ok = None
        if bc_ok is None and bc_res:
            try:
                bc_ok = all(sp.simplify(item) == 0 for item in bc_res)
            except Exception:
                bc_ok = None
        required = []
        if require_pde:
            required.append(pde_ok)
        if require_initial:
            required.append(init_ok)
        if require_boundary:
            required.append(bc_ok)
        if required:
            if any(bit is False for bit in required):
                verified = False
            elif all(bit is True for bit in required):
                verified = True
            else:
                verified = None
        elif verified is None:
            bits = [bit for bit in (pde_ok, init_ok, bc_ok) if bit is not None]
            verified = all(bits) if bits else None
        if required:
            status = (
                "verified"
                if verified is True
                else "failed"
                if verified is False
                else "unverified"
            )
        else:
            status = ver.get("status") or (
                "verified"
                if verified is True
                else "failed"
                if verified is False
                else "unverified"
            )
        return PDEVerificationSummary(
            verified=verified,
            status=status,
            pde_verified=pde_ok,
            initial_verified=init_ok,
            boundary_verified=bc_ok,
            pde_residual=pde_res,
            initial_residuals=init_res,
            boundary_residuals=bc_res,
            mode=ver.get("mode", "unknown"),
            message=ver.get("message", ver.get("error", "")),
        )
    return PDEVerificationSummary(
        None, "unverified", mode="unknown", message="" if ver is None else str(ver)
    )


def append_trace_step(
    steps: list[PDESolverTraceStep],
    method: str,
    stage: str,
    success: bool,
    message: str = "",
    exception: Exception | None = None,
    verification: Any = None,
) -> None:
    steps.append(
        PDESolverTraceStep(
            method=method,
            stage=stage,
            success=bool(success),
            message=str(message),
            exception_type=type(exception).__name__ if exception is not None else None,
            verification=as_verification_summary(verification)
            if verification is not None
            else None,
        )
    )


def coerce_result(
    raw_result: Any,
    *,
    attempted_methods: list[str],
    trace_steps: list[PDESolverTraceStep],
    canonical_eq,
    assumptions,
    dep_expr_or_func,
    indep_vars,
    ics=None,
    bcs=None,
    normalize_result=True,
    normalization_max_ops=80,
) -> PDESolutionRecord:
    verify_opts = {
        "require_pde": canonical_eq is not None,
        "require_initial": ics is not None,
        "require_boundary": bcs is not None,
    }

    def summarize(value):
        return as_verification_summary(value, **verify_opts)

    if isinstance(raw_result, PDESolutionRecord):
        record = raw_result
    elif isinstance(raw_result, BasePDEResult):
        raw_verification = getattr(raw_result, "verification", None)
        if raw_verification:
            verification = summarize(raw_verification)
        elif raw_result.method in {
            "quasilinear_implicit_characteristics",
            "scalar_conservation_riemann_shock",
            "scalar_conservation_riemann_rarefaction",
        }:
            verification = summarize(
                {
                    "verified": None,
                    "status": "unverified",
                    "mode": "method_specific",
                    "message": "skipped expensive generic verification for implicit/weak first-order solution",
                }
            )
        else:
            verification = summarize(
                verify_result(
                    canonical_eq,
                    raw_result,
                    dep_expr_or_func,
                    indep_vars,
                    ics=ics,
                    bcs=bcs,
                    assumptions=assumptions,
                )
            )
        record = PDESolutionRecord(
            method=raw_result.method,
            solution=raw_result.solution,
            classification=getattr(raw_result, "classification", None),
            steps=tuple(attempted_methods),
            verification=verification.as_dict(),
            assumptions=getattr(raw_result, "assumptions", assumptions),
            metadata=dict(getattr(raw_result, "metadata", {}) or {}),
            reduced_problem=getattr(raw_result, "reduced_problem", None),
            warnings=tuple(getattr(raw_result, "warnings", ()) or ()),
            canonical_equation=canonical_eq,
        )
    else:
        solution = getattr(raw_result, "solution", raw_result)
        method = getattr(raw_result, "method", "unknown")
        details = dict(getattr(raw_result, "details", {}) or {})
        details.setdefault("raw_result", raw_result)
        fam = getattr(raw_result, "method_family", None)
        if fam is not None:
            details.setdefault("method_family", fam)
        raw_verification = getattr(raw_result, "verification", None)
        if raw_verification:
            if (
                isinstance(raw_verification, (list, tuple))
                and raw_verification
                and isinstance(raw_verification[0], dict)
            ):
                first = raw_verification[0]
                verification = summarize(
                    {
                        "verified": first.get("verified"),
                        "pde_verified": first.get("verified"),
                        "pde_residual": first.get("residual"),
                        "mode": "method_specific",
                        "message": "used method-specific verification",
                    }
                )
            else:
                verification = summarize(raw_verification)
        elif method in {
            "quasilinear_implicit_characteristics",
            "scalar_conservation_riemann_shock",
            "scalar_conservation_riemann_rarefaction",
        }:
            verification = summarize(
                {
                    "verified": None,
                    "status": "unverified",
                    "mode": "method_specific",
                    "message": "skipped expensive generic verification for implicit/weak first-order solution",
                }
            )
        else:
            verification = summarize(
                verify_result(
                    canonical_eq,
                    raw_result,
                    dep_expr_or_func,
                    indep_vars,
                    ics=ics,
                    bcs=bcs,
                    assumptions=assumptions,
                )
            )
        record = PDESolutionRecord(
            method=method,
            solution=solution,
            classification=details.get("classification"),
            steps=tuple(attempted_methods),
            verification=verification.as_dict(),
            assumptions=assumptions,
            metadata=dict(details),
            reduced_problem=details.get("reduced_problem"),
            warnings=tuple(details.get("warnings", ()) or ()),
            canonical_equation=canonical_eq,
        )
    normalized_solution, normalization_report = normalize_solution(
        record.solution,
        method=record.method,
        policy=NormalizationPolicy(
            enabled=bool(normalize_result), max_ops=int(normalization_max_ops)
        ),
    )
    if normalized_solution != record.solution:
        record = PDESolutionRecord(
            method=record.method,
            solution=normalized_solution,
            classification=getattr(record, "classification", None),
            steps=record.steps,
            verification=record.verification,
            assumptions=record.assumptions,
            metadata=dict(getattr(record, "metadata", {}) or {}),
            reduced_problem=getattr(record, "reduced_problem", None),
            warnings=tuple(getattr(record, "warnings", ()) or ()),
            canonical_equation=record.canonical_equation,
        )
    trace = PDEExecutionTrace(
        selected_method=record.method,
        attempted_methods=tuple(attempted_methods),
        steps=tuple(trace_steps),
    )
    metadata = dict(record.metadata)
    metadata["normalization"] = normalization_report.as_dict()
    metadata["trace"] = trace
    metadata.setdefault("attempted_methods", tuple(attempted_methods))
    ver = summarize(record.verification)
    return PDESolutionRecord(
        method=record.method,
        solution=record.solution,
        classification=getattr(record, "classification", None),
        steps=record.steps,
        verification=ver.as_dict(),
        assumptions=record.assumptions,
        metadata=metadata,
        reduced_problem=getattr(record, "reduced_problem", None),
        warnings=tuple(getattr(record, "warnings", ()) or ()),
        canonical_equation=record.canonical_equation,
    )


__all__ = ["as_verification_summary", "append_trace_step", "coerce_result"]
