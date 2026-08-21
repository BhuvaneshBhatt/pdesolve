from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import sympy as sp


@dataclass(frozen=True, kw_only=True)
class SolverMethodResult:
    """Shared lightweight result shape for method-specific solver modules."""

    method_family: str
    solution: Any
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def method(self) -> str:
        return self.method_family


@dataclass(frozen=True)
class PDEVerificationSummary:
    verified: bool | None
    status: str
    pde_verified: bool | None = None
    initial_verified: bool | None = None
    boundary_verified: bool | None = None
    pde_residual: Any = None
    initial_residuals: tuple[Any, ...] = ()
    boundary_residuals: tuple[Any, ...] = ()
    mode: str = "unknown"
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "status": self.status,
            "pde_verified": self.pde_verified,
            "initial_verified": self.initial_verified,
            "boundary_verified": self.boundary_verified,
            "pde_residual": self.pde_residual,
            "initial_residuals": self.initial_residuals,
            "boundary_residuals": self.boundary_residuals,
            "mode": self.mode,
            "message": self.message,
        }


@dataclass(frozen=True)
class PDESolverTraceStep:
    method: str
    stage: str
    success: bool
    message: str = ""
    exception_type: str | None = None
    verification: PDEVerificationSummary | None = None


@dataclass(frozen=True)
class PDEExecutionTrace:
    selected_method: str | None
    attempted_methods: tuple[str, ...]
    steps: tuple[PDESolverTraceStep, ...] = ()


@dataclass(frozen=True)
class CanonicalPDERepresentation:
    dependent_variables: tuple[sp.Expr, ...]
    independent_variables: tuple[sp.Symbol, ...]
    normalized_equations: tuple[sp.Equality, ...]
    order: int
    linearity: str
    principal_part: Any = None
    coefficient_dependence: tuple[str, ...] = ()
    principal_multiindex: Any = None
    ic_metadata: dict[str, Any] = field(default_factory=dict)
    bc_metadata: dict[str, Any] = field(default_factory=dict)
    domain_metadata: dict[str, Any] = field(default_factory=dict)
    geometry_metadata: dict[str, Any] = field(default_factory=dict)
    time_slice_metadata: dict[str, Any] = field(default_factory=dict)
    weak_solution_flags: tuple[str, ...] = ()
    transformability_tags: tuple[str, ...] = ()
    recognized_tags: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PDEProblemProfile:
    indep_vars: tuple[sp.Symbol, ...]
    dep_function: sp.Expr
    normalized_equation: sp.Equality
    zero_expression: sp.Expr
    order: int
    principal_solved_form: object | None
    characteristic_data: object | None
    first_order_linear: object | None
    second_order_class: object | None
    canonical_family: str | None
    conservation_law: object | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PDESolverMethodCandidate:
    method: str
    score: int
    reasons: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PDESolutionPlan:
    profile: PDEProblemProfile
    steps: tuple[PDESolverMethodCandidate, ...]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BasePDEResult:
    method: str
    solution: Any
    classification: Any = None
    assumptions: Any = True
    verification: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    reduced_problem: Any = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "verification", MappingProxyType(dict(self.verification or {})))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))


@dataclass(frozen=True)
class ClosedFormPDEResult(BasePDEResult):
    pass


@dataclass(frozen=True)
class ImplicitPDEResult(BasePDEResult):
    pass


@dataclass(frozen=True)
class SeriesPDEResult(BasePDEResult):
    series_terms: Any = None


@dataclass(frozen=True)
class TransformPDEResult(BasePDEResult):
    transform_data: Any = None


@dataclass(frozen=True)
class WeakSolutionResult(BasePDEResult):
    admissibility: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EigenfunctionExpansionResult(SeriesPDEResult):
    eigen_data: Any = None


@dataclass(frozen=True)
class SystemPDEResult(BasePDEResult):
    system_size: int = 1
    transform: Any = None
    characteristic_variables: Any = None

    @property
    def details(self):
        return self.metadata


@dataclass(frozen=True)
class KernelRepresentationResult(BasePDEResult):
    kernel: Any = None
    source_point: Any = None
    operator_family: str | None = None


@dataclass(frozen=True)
class FundamentalSolutionResult(KernelRepresentationResult):
    pass


@dataclass(frozen=True)
class GreenFunctionResult(KernelRepresentationResult):
    boundary_type: str | None = None


@dataclass(frozen=True)
class NumericalFallbackResult(BasePDEResult):
    numerical_object: Any = None


@dataclass(frozen=True)
class UnsolvedButReducedResult(BasePDEResult):
    pass


@dataclass(frozen=True)
class PDESolutionRecord(ClosedFormPDEResult):
    steps: tuple[str, ...] = ()
    canonical_equation: sp.Equality | None = None

    @property
    def details(self):
        """Structured solver metadata for the returned solution."""
        return self.metadata

    @property
    def verified(self) -> bool | None:
        return self.verification.get("verified")

    @property
    def status(self) -> str | None:
        return self.verification.get("status")


@dataclass(frozen=True)
class ConservationLawCanonicalForm:
    indep_vars: tuple[sp.Symbol, sp.Symbol]
    dep_function: sp.Expr
    density: sp.Expr
    flux: sp.Expr
    source: sp.Expr = sp.Integer(0)
    normalized_equation: sp.Equality | None = None
    autonomous_flux: sp.Expr | None = None
    family: str = "scalar_conservation_law"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def characteristic_speed(self):
        if self.autonomous_flux is None:
            return None
        usym = sp.Symbol("u", real=True)
        speed = sp.simplify(sp.diff(self.autonomous_flux, usym))
        if speed.free_symbols == {usym}:
            return None
        return speed


@dataclass(frozen=True)
class ConservationLawInitialData1D:
    indep_vars: tuple[sp.Symbol, sp.Symbol]
    dep_function: sp.Expr
    kind: str
    profile: Any = None
    equation: sp.Equality | None = None
    left_state: Any = None
    right_state: Any = None
    interface: Any = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConservationLawPropagationResult:
    method: str
    solution: sp.Equality
    profile: Any
    speed: Any
    canonical_form: ConservationLawCanonicalForm | None = None
    initial_data: ConservationLawInitialData1D | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConservationLawImplicitCharacteristicResult:
    method: str
    solution: sp.Equality | tuple[sp.Equality, ...]
    profile: Any
    characteristic_parameter: Any
    characteristic_relation: sp.Equality
    profile_relation: sp.Equality
    footpoint_equation: sp.Equality | None = None
    implicit_relation: sp.Equality | None = None
    characteristic_speed: Any = None
    canonical_form: ConservationLawCanonicalForm | None = None
    initial_data: ConservationLawInitialData1D | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConservationLawShockResult:
    method: str
    solution: sp.Equality
    flux: sp.Expr
    left_state: Any
    right_state: Any
    shock_speed: Any
    canonical_form: ConservationLawCanonicalForm | None = None
    initial_data: ConservationLawInitialData1D | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConservationLawRarefactionResult:
    method: str
    solution: sp.Equality
    flux: sp.Expr
    left_state: Any
    right_state: Any
    left_speed: Any
    right_speed: Any
    self_similar_variable: Any
    canonical_form: ConservationLawCanonicalForm | None = None
    initial_data: ConservationLawInitialData1D | None = None
    details: dict[str, Any] = field(default_factory=dict)
