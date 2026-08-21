"""Public API for :mod:`pdesolve`."""

from importlib.metadata import version as _dist_version

__version__ = _dist_version("pdesolve")
from .benchmark_suite import (
    BenchmarkCase as BenchmarkCase,
)
from .benchmark_suite import (
    BenchmarkOutcome as BenchmarkOutcome,
)
from .benchmark_suite import (
    BenchmarkSuite as BenchmarkSuite,
)
from .benchmark_suite import (
    build_benchmark_suite as build_benchmark_suite,
)
from .benchmark_suite import (
    run_benchmark_case as run_benchmark_case,
)
from .benchmark_suite import (
    run_benchmark_suite as run_benchmark_suite,
)
from .boundary_model import (
    BoundaryComponent,
    BoundaryConditionBinding,
    BoundaryModel,
    build_boundary_model,
)
from .classical_methods import (
    PDEIVPResult,
    solve_euler_bernoulli_beam_freefree_ibvp,
    solve_linear_constant_coefficient_pde_bvp_2d,
)
from .complete_integral_helpers import (
    CompleteIntegralResult,
    fit_complete_integral_to_initial_curve,
    integrate_pfaffian_equation,
    process_initial_curve_2d,
    solve_charpit_complete_integral_2vars,
    solve_complete_integral_pde,
    solve_first_order_cauchy_problem_2d,
    solve_jacobi_complete_integral,
)
from .conditions import ConditionEquation, ConditionModel, parse_conditions
from .conservation_laws import (
    ConservationLawCanonicalForm,
    ConservationLawImplicitCharacteristicResult,
    ConservationLawInitialData1D,
    ConservationLawPropagationResult,
    ConservationLawRarefactionResult,
    ConservationLawShockResult,
    analyze_conservation_law,
    canonicalize_scalar_conservation_law,
    canonicalize_scalar_conservation_law_1d,
    parse_conservation_law_initial_data,
    parse_scalar_conservation_law_initial_data,
    solve_scalar_conservation_law_ivp,
    verify_conservation_law_solution,
    verify_piecewise_conservation_law_solution,
    verify_weak_conservation_law_solution,
)
from .dispatcher import extract_solution_trace, pdesolve, summarize_solution_record
from .domains import (
    DiskDomain,
    DomainGeometry,
    FullLineDomain,
    HalfLineDomain,
    IntervalDomain,
    PolarAnnulusDomain,
    RectangleDomain,
    infer_domain_geometry,
)
from .errors import (
    PDEError,
    PDEInputError,
    PDEMethodNotApplicable,
    PDESolveError,
    PDETransformationError,
    PDEVerificationError,
)
from .first_order_api import solve_first_order_pde
from .first_order_framework import CanonicalFirstOrderPDE, canonicalize_first_order_nonlinear_pde
from .first_order_geometry import (
    AdaptedCoordinateReduction,
    adapted_coordinate_reduction,
    characteristic_first_integral,
)
from .first_order_linear import (
    FirstOrderPDEResult,
    LinearFirstOrderProfile,
    parse_linear_first_order,
    recognize_first_order_linear_pde,
    solve_first_order_linear_pde,
)
from .first_order_nonlinear import (
    ConstantCharacteristicProfile,
    FirstOrderNonlinearAnalysis,
    InvariantReductionCandidate,
    analyze_first_order_nonlinear_pde,
    enumerate_invariant_reduction_candidates,
    recognize_const_characteristics,
    solve_first_order_nonlinear_auto,
    solve_first_order_quasilinear_pde,
    solve_via_invariant_reduction,
)
from .green_subsystem import (
    AdvancedGreenPlan,
    execute_advanced_green_plan,
    recognize_advanced_kernel_problem,
    solve_linear_ode_green_function,
)
from .hyperbolic_system import (
    CanonicalLinearSystemPDE,
    HyperbolicSystemResult,
    extract_canonical_linear_system_form,
    solve_hyperbolic_system,
)
from .kernels import (
    KernelMethodPlan,
    build_kernel_method_plan,
    execute_kernel_plan,
    solve_fundamental_solution,
    solve_green_function,
)
from .lie_analysis import (
    LieInvariantCoordinates,
    LiePointSymmetryAnalysis,
    analyze_lie_point_symmetries,
    invariants_from_point_generator,
)
from .method_names import normalize_method_name
from .planners import plan_canonical_problem
from .problem import PDEProblem, build_pde_problem
from .recognition import (
    PDERecognitionRecord,
    build_canonical_representation,
    recognize_pde_structure,
)
from .recognizers import recognize_canonical_problem
from .result_verification import (
    ClassicalResidualVerifier,
    ImplicitSolutionVerifier,
    KernelVerifier,
    SeriesVerifier,
    TransformVerifier,
    VerificationStrategy,
    WeakSolutionVerifier,
    select_verification_strategy,
    verify_result,
)
from .results import (
    BasePDEResult,
    CanonicalPDERepresentation,
    ClosedFormPDEResult,
    EigenfunctionExpansionResult,
    FundamentalSolutionResult,
    GreenFunctionResult,
    ImplicitPDEResult,
    KernelRepresentationResult,
    NumericalFallbackResult,
    PDEExecutionTrace,
    PDEProblemProfile,
    PDESolutionPlan,
    PDESolutionRecord,
    PDESolverMethodCandidate,
    PDESolverTraceStep,
    PDEVerificationSummary,
    SeriesPDEResult,
    SolverMethodResult,
    SystemPDEResult,
    TransformPDEResult,
    UnsolvedButReducedResult,
    WeakSolutionResult,
)
from .separation_framework import SeparableGeometryPlan, build_separable_geometry_plan
from .separation_general import ProductSeparationResult, separate_product_pde
from .solver_execution import solve_with_canonical_problem
from .solvers import execute_planned_solver
from .special_pdes import (
    SpecialPDEResult,
    recognize_heat_or_advection_diffusion,
    recognize_laplace_or_helmholtz,
    solve_special_pde,
)
from .sturm_liouville import (
    SturmLiouvilleProblem,
    SturmLiouvilleSpectrum,
    solve_regular_constant_sturm_liouville,
)
from .transform_framework import TransformMethodPlan, build_transform_method_plan
from .transform_postprocess import (
    TransformPostprocessReport,
    evaluate_inner_transforms,
    postprocess_transform_result,
)
from .unified_transform import (
    EvolutionPDEProfile,
    UnifiedTransformResult,
    determine_dispersion_relation,
    recognize_evolution_pde,
    solve_unified_transform,
    solve_unified_transform_half_line,
    solve_unified_transform_whole_line,
)

__all__ = [
    "__version__",
    "PDEError",
    "PDEInputError",
    "PDEMethodNotApplicable",
    "PDETransformationError",
    "PDEVerificationError",
    "PDESolveError",
    "pdesolve",
    "extract_solution_trace",
    "summarize_solution_record",
    "PDEProblem",
    "build_pde_problem",
    "SolverMethodResult",
    "PDEVerificationSummary",
    "PDESolverTraceStep",
    "PDEExecutionTrace",
    "CanonicalPDERepresentation",
    "PDEProblemProfile",
    "PDESolverMethodCandidate",
    "PDESolutionPlan",
    "BasePDEResult",
    "ClosedFormPDEResult",
    "ImplicitPDEResult",
    "SeriesPDEResult",
    "TransformPDEResult",
    "WeakSolutionResult",
    "EigenfunctionExpansionResult",
    "SystemPDEResult",
    "NumericalFallbackResult",
    "UnsolvedButReducedResult",
    "PDESolutionRecord",
    "PDERecognitionRecord",
    "build_canonical_representation",
    "recognize_pde_structure",
    "AdaptedCoordinateReduction",
    "adapted_coordinate_reduction",
    "characteristic_first_integral",
    "FirstOrderPDEResult",
    "LinearFirstOrderProfile",
    "parse_linear_first_order",
    "recognize_first_order_linear_pde",
    "solve_first_order_linear_pde",
    "ConstantCharacteristicProfile",
    "FirstOrderNonlinearAnalysis",
    "InvariantReductionCandidate",
    "analyze_first_order_nonlinear_pde",
    "enumerate_invariant_reduction_candidates",
    "recognize_const_characteristics",
    "solve_first_order_quasilinear_pde",
    "solve_via_invariant_reduction",
    "solve_first_order_nonlinear_auto",
    "solve_first_order_pde",
    "SpecialPDEResult",
    "recognize_heat_or_advection_diffusion",
    "recognize_laplace_or_helmholtz",
    "solve_special_pde",
    "EvolutionPDEProfile",
    "UnifiedTransformResult",
    "determine_dispersion_relation",
    "recognize_evolution_pde",
    "solve_unified_transform",
    "solve_unified_transform_half_line",
    "solve_unified_transform_whole_line",
    "HyperbolicSystemResult",
    "solve_hyperbolic_system",
    "normalize_method_name",
    "integrate_pfaffian_equation",
    "solve_charpit_complete_integral_2vars",
    "solve_jacobi_complete_integral",
    "solve_complete_integral_pde",
    "solve_linear_constant_coefficient_pde_bvp_2d",
    "solve_euler_bernoulli_beam_freefree_ibvp",
    "CompleteIntegralResult",
    "PDEIVPResult",
    "process_initial_curve_2d",
    "fit_complete_integral_to_initial_curve",
    "solve_first_order_cauchy_problem_2d",
    "canonicalize_scalar_conservation_law_1d",
    "parse_scalar_conservation_law_initial_data",
    "solve_scalar_conservation_law_ivp",
    "verify_piecewise_conservation_law_solution",
    "verify_weak_conservation_law_solution",
    "canonicalize_scalar_conservation_law",
    "parse_conservation_law_initial_data",
    "analyze_conservation_law",
    "verify_conservation_law_solution",
    "ConservationLawCanonicalForm",
    "ConservationLawInitialData1D",
    "ConservationLawPropagationResult",
    "ConservationLawImplicitCharacteristicResult",
    "ConservationLawShockResult",
    "ConservationLawRarefactionResult",
    "ConditionEquation",
    "ConditionModel",
    "parse_conditions",
    "DomainGeometry",
    "IntervalDomain",
    "RectangleDomain",
    "DiskDomain",
    "HalfLineDomain",
    "FullLineDomain",
    "PolarAnnulusDomain",
    "infer_domain_geometry",
    "CanonicalLinearSystemPDE",
    "extract_canonical_linear_system_form",
    "SeparableGeometryPlan",
    "build_separable_geometry_plan",
    "TransformMethodPlan",
    "build_transform_method_plan",
    "BoundaryComponent",
    "BoundaryConditionBinding",
    "BoundaryModel",
    "build_boundary_model",
    "CanonicalFirstOrderPDE",
    "canonicalize_first_order_nonlinear_pde",
    "BenchmarkCase",
    "BenchmarkOutcome",
    "BenchmarkSuite",
    "build_benchmark_suite",
    "run_benchmark_case",
    "run_benchmark_suite",
]


__all__ += [
    "solve_with_canonical_problem",
    "recognize_canonical_problem",
    "plan_canonical_problem",
    "execute_planned_solver",
]


__all__ += [
    "KernelRepresentationResult",
    "FundamentalSolutionResult",
    "GreenFunctionResult",
    "KernelMethodPlan",
    "build_kernel_method_plan",
    "solve_fundamental_solution",
    "solve_green_function",
    "execute_kernel_plan",
]


__all__ += [
    "AdvancedGreenPlan",
    "recognize_advanced_kernel_problem",
    "execute_advanced_green_plan",
    "solve_linear_ode_green_function",
]


__all__ += [
    "VerificationStrategy",
    "ClassicalResidualVerifier",
    "ImplicitSolutionVerifier",
    "WeakSolutionVerifier",
    "KernelVerifier",
    "SeriesVerifier",
    "TransformVerifier",
    "select_verification_strategy",
    "verify_result",
    "ProductSeparationResult",
    "separate_product_pde",
    "SturmLiouvilleProblem",
    "SturmLiouvilleSpectrum",
    "solve_regular_constant_sturm_liouville",
    "TransformPostprocessReport",
    "evaluate_inner_transforms",
    "postprocess_transform_result",
    "LiePointSymmetryAnalysis",
    "analyze_lie_point_symmetries",
    "LieInvariantCoordinates",
    "invariants_from_point_generator",
]
