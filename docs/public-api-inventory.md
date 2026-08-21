# Public API inventory

`pdesolve.__all__` contains **152 package-level exports**. The main solving entry point is `pdesolve(...)`; the remaining exports provide focused solvers, structured models, planning, verification, and inspection APIs.

## `pdesolve`

| Export | Kind | Signature |
|---|---|---|
| `__version__` | str | `` |

## `pdesolve.boundary_model`

| Export | Kind | Signature |
|---|---|---|
| `BoundaryComponent` | class | `(name: 'str', variable: 'sp.Symbol \| None', location: 'sp.Expr \| None', geometry_kind: 'str', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `BoundaryConditionBinding` | class | `(component: 'BoundaryComponent', kind: 'str', equation: 'sp.Equality', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `BoundaryModel` | class | `(geometry: 'DomainGeometry \| None', components: 'tuple[BoundaryComponent, ...]' = (), bindings: 'tuple[BoundaryConditionBinding, ...]' = (), metadata: 'dict[str, Any]' = <factory>) -> None` |
| `build_boundary_model` | function | `(geometry: 'DomainGeometry \| None', conditions: 'ConditionModel \| None') -> 'BoundaryModel \| None'` |

## `pdesolve.classical_first_order`

| Export | Kind | Signature |
|---|---|---|
| `PDEIVPResult` | class | `(method: 'str', solution: 'sp.Expr \| sp.Equality', details: 'dict') -> None` |

## `pdesolve.classical_methods`

| Export | Kind | Signature |
|---|---|---|
| `solve_euler_bernoulli_beam_freefree_ibvp` | function | `(dep_expr_or_func, *, x, t, length, stiffness=1, initial_displacement=None, initial_velocity=None, n_terms=8)` |
| `solve_linear_constant_coefficient_pde_bvp_2d` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None, assumptions=True)` |

## `pdesolve.complete_integral_helpers`

| Export | Kind | Signature |
|---|---|---|
| `CompleteIntegralResult` | class | `(method: 'str', solutions: 'tuple[sp.Equality, ...]', details: 'dict', verification: 'tuple[dict, ...]' = ()) -> None` |
| `fit_complete_integral_to_initial_curve` | function | `(solution_eq, initial, dep_expr_or_func, indep_vars=None)` |
| `integrate_pfaffian_equation` | function | `(p_fields, dep_expr_or_func, indep_vars=None, *, dependent_symbol=None, allow_implicit=True, constant_symbol=None)` |
| `process_initial_curve_2d` | function | `(initial, dep_expr_or_func, indep_vars=None, *, parameter=None)` |
| `solve_charpit_complete_integral_2vars` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, parameter_symbol=None)` |
| `solve_complete_integral_pde` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True, **kwargs)` |
| `solve_first_order_cauchy_problem_2d` | function | `(eq_or_expr, initial, dep_expr_or_func, indep_vars=None, *, assumptions=True, **kwargs)` |
| `solve_jacobi_complete_integral` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, parameter_symbols=None)` |

## `pdesolve.conditions`

| Export | Kind | Signature |
|---|---|---|
| `ConditionEquation` | class | `(equation: 'sp.Equality', role: 'str', variable: 'sp.Symbol \| None' = None, location: 'sp.Expr \| None' = None, derivative_multiindex: 'tuple[int, ...]' = (), metadata: 'dict[str, Any]' = <factory>, kind: 'str \| None' = None) -> None` |
| `ConditionModel` | class | `(dependent_function: 'Any', independent_variables: 'tuple[sp.Symbol, ...]', initial_conditions: 'tuple[ConditionEquation, ...]' = (), boundary_conditions: 'tuple[ConditionEquation, ...]' = (), mixed_conditions: 'tuple[ConditionEquation, ...]' = (), event_conditions: 'tuple[ConditionEquation, ...]' = (), metadata: 'dict[str, Any]' = <factory>) -> None` |
| `parse_conditions` | function | `(ics=None, bcs=None, *, dep_expr=None, indep_vars: 'tuple[sp.Symbol, ...]' = ()) -> 'ConditionModel'` |

## `pdesolve.conservation_laws`

| Export | Kind | Signature |
|---|---|---|
| `analyze_conservation_law` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None)` |
| `canonicalize_scalar_conservation_law` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None)` |
| `canonicalize_scalar_conservation_law_1d` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None)` |
| `parse_conservation_law_initial_data` | function | `(initial_conditions, dep_expr_or_func, indep_vars=None)` |
| `parse_scalar_conservation_law_initial_data` | function | `(initial_conditions, dep_expr_or_func, indep_vars=None)` |
| `solve_scalar_conservation_law_ivp` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, initial_conditions=None)` |
| `verify_conservation_law_solution` | function | `(eq_or_expr, solution, dep_expr_or_func, indep_vars=None, *, structured_result=None, initial_conditions=None)` |
| `verify_piecewise_conservation_law_solution` | function | `(eq_or_expr, solution, dep_expr_or_func, indep_vars=None, *, initial_conditions=None)` |
| `verify_weak_conservation_law_solution` | function | `(solution, dep_expr_or_func=None, indep_vars=None)` |

## `pdesolve.dispatcher`

| Export | Kind | Signature |
|---|---|---|
| `extract_solution_trace` | function | `(record)` |
| `pdesolve` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None, method='auto', assumptions=True, **kwargs)` |
| `summarize_solution_record` | function | `(record)` |

## `pdesolve.domains`

| Export | Kind | Signature |
|---|---|---|
| `DiskDomain` | class | `(kind: 'str', coordinates: 'tuple[sp.Symbol, ...]', extents: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `DomainGeometry` | class | `(kind: 'str', coordinates: 'tuple[sp.Symbol, ...]', extents: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `FullLineDomain` | class | `(kind: 'str', coordinates: 'tuple[sp.Symbol, ...]', extents: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `HalfLineDomain` | class | `(kind: 'str', coordinates: 'tuple[sp.Symbol, ...]', extents: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `IntervalDomain` | class | `(kind: 'str', coordinates: 'tuple[sp.Symbol, ...]', extents: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `PolarAnnulusDomain` | class | `(kind: 'str', coordinates: 'tuple[sp.Symbol, ...]', extents: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `RectangleDomain` | class | `(kind: 'str', coordinates: 'tuple[sp.Symbol, ...]', extents: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>) -> None` |
| `infer_domain_geometry` | function | `(*, indep_vars: 'tuple[sp.Symbol, ...]', bcs=None, condition_model=None, domain=None, assumptions=True) -> 'DomainGeometry'` |

## `pdesolve.errors`

| Export | Kind | Signature |
|---|---|---|
| `PDEError` | class | `` |
| `PDEInputError` | class | `` |
| `PDEMethodNotApplicable` | class | `` |
| `PDESolveError` | class | `` |
| `PDETransformationError` | class | `` |
| `PDEVerificationError` | class | `` |

## `pdesolve.first_order_api`

| Export | Kind | Signature |
|---|---|---|
| `solve_first_order_pde` | function | `(eq, u, vars)` |

## `pdesolve.first_order_framework`

| Export | Kind | Signature |
|---|---|---|
| `CanonicalFirstOrderPDE` | class | `(variables: 'tuple[sp.Symbol, ...]', dependent_variable: 'sp.Expr', equation: 'sp.Equality', F: 'sp.Expr', p_symbol: 'sp.Symbol', q_symbol: 'sp.Symbol \| None', recognized_family: 'str', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `canonicalize_first_order_nonlinear_pde` | function | `(eq, uexpr, vars_)` |

## `pdesolve.first_order_geometry`

| Export | Kind | Signature |
|---|---|---|
| `AdaptedCoordinateReduction` | class | `(invariant: 'sp.Expr', transverse_var: 'sp.Symbol', param: 'sp.Symbol', subst_map: 'dict[sp.Symbol, sp.Expr]', coeff: 'sp.Expr') -> None` |
| `adapted_coordinate_reduction` | function | `(a: 'sp.Expr', b: 'sp.Expr', x: 'sp.Symbol', y: 'sp.Symbol') -> 'Optional[AdaptedCoordinateReduction]'` |
| `characteristic_first_integral` | function | `(a: 'sp.Expr', b: 'sp.Expr', x: 'sp.Symbol', y: 'sp.Symbol') -> 'Optional[sp.Expr]'` |

## `pdesolve.first_order_linear`

| Export | Kind | Signature |
|---|---|---|
| `FirstOrderPDEResult` | class | `(*, method_family: 'str', solution: 'Any', details: 'dict[str, Any]' = <factory>, invariant: 'Optional[sp.Expr]' = None, reduction: 'Optional[AdaptedCoordinateReduction]' = None) -> None` |
| `LinearFirstOrderProfile` | class | `(a: 'sp.Expr', b: 'sp.Expr', c: 'sp.Expr', d: 'sp.Expr', reduction: 'Optional[AdaptedCoordinateReduction]') -> None` |
| `parse_linear_first_order` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', x: 'sp.Symbol', y: 'sp.Symbol') -> 'LinearFirstOrderProfile'` |
| `recognize_first_order_linear_pde` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'Optional[LinearFirstOrderProfile]'` |
| `solve_first_order_linear_pde` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'FirstOrderPDEResult'` |

## `pdesolve.first_order_nonlinear`

| Export | Kind | Signature |
|---|---|---|
| `ConstantCharacteristicProfile` | class | `(a: 'sp.Expr', b: 'sp.Expr', source: 'sp.Expr', invariant: 'sp.Expr') -> None` |
| `FirstOrderNonlinearAnalysis` | class | `(is_first_order: 'bool', is_linear_first_order: 'bool', is_quasilinear: 'bool', conservation_law_family: 'str \| None', burgers_family: 'str \| None', recommended_methods: 'tuple[str, ...]', details: 'dict[str, Any]') -> None` |
| `InvariantReductionCandidate` | class | `(method: 'str', score: 'int', invariants: 'tuple[sp.Expr, ...]', transverse_params: 'tuple[sp.Expr, ...]', reduced_eq: 'sp.Equality', ansatz: 'sp.Expr', red_func: 'Any' = None, details: 'dict[str, Any] \| None' = None) -> None` |
| `analyze_first_order_nonlinear_pde` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None)` |
| `enumerate_invariant_reduction_candidates` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, degree=1, max_subset_size=2, max_degree=2)` |
| `recognize_const_characteristics` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'Optional[ConstantCharacteristicProfile]'` |
| `solve_first_order_nonlinear_auto` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None, assumptions=True, **kwargs)` |
| `solve_first_order_quasilinear_pde` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'FirstOrderPDEResult'` |
| `solve_via_invariant_reduction` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None, assumptions=True, degree=1, max_subset_size=2, max_symmetry_steps=2, max_degree=2)` |

## `pdesolve.green_subsystem`

| Export | Kind | Signature |
|---|---|---|
| `AdvancedGreenPlan` | class | `(method: 'str', operator_family: 'str', geometry_kind: 'str', boundary_family: 'str', source_point: 'tuple[sp.Expr, ...]', metadata: 'dict[str, Any]') -> None` |
| `execute_advanced_green_plan` | function | `(eq_or_expr, dep_expr_or_func, indep_vars, *, bcs=None, geometry=None, assumptions=True)` |
| `recognize_advanced_kernel_problem` | function | `(eq_or_expr, dep_expr_or_func, indep_vars)` |
| `solve_linear_ode_green_function` | function | `(eq_or_expr, dep_expr_or_func, var, *, conditions=None)` |

## `pdesolve.hyperbolic_system`

| Export | Kind | Signature |
|---|---|---|
| `CanonicalLinearSystemPDE` | class | `(equations: 'tuple[sp.Equality, ...]', variables: 'tuple[sp.Symbol, sp.Symbol]', unknowns: 'tuple[sp.FunctionClass, ...]', coeff_matrix: 'sp.Matrix', forcing: 'sp.Matrix', variable_coefficient: 'bool', diagonalizable: 'bool', transform_matrix: 'sp.Matrix \| None' = None, diagonal_matrix: 'sp.Matrix \| None' = None) -> None` |
| `HyperbolicSystemResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), system_size: 'int' = 1, transform: 'Any' = None, *, characteristic_variables: 'tuple[sp.Expr, ...]' = (), solution_map: 'dict[sp.Expr, sp.Expr]', coeff_matrix: 'sp.Matrix', eigenvalues: 'tuple[sp.Expr, ...]', canonical_system: 'CanonicalLinearSystemPDE \| None' = None) -> None` |
| `extract_canonical_linear_system_form` | function | `(eqns: 'Sequence[sp.Equality \| sp.Expr]', funcs: 'Sequence[sp.Function]', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'CanonicalLinearSystemPDE'` |
| `solve_hyperbolic_system` | function | `(eqns: 'Sequence[sp.Equality \| sp.Expr]', ics: 'Sequence[sp.Equality]', funcs: 'Sequence[sp.Function]', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'HyperbolicSystemResult'` |

## `pdesolve.kernels`

| Export | Kind | Signature |
|---|---|---|
| `KernelMethodPlan` | class | `(method: 'str', operator_family: 'str', geometry_kind: 'str', boundary_family: 'str', source_point: 'tuple[sp.Expr, ...]', metadata: 'dict[str, Any]') -> None` |
| `build_kernel_method_plan` | function | `(problem, *, geometry=None)` |
| `execute_kernel_plan` | function | `(problem, *, plan: 'KernelMethodPlan \| None' = None)` |
| `solve_fundamental_solution` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True, source_point=None)` |
| `solve_green_function` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, bcs=None, assumptions=True, geometry=None, source_point=None)` |

## `pdesolve.lie_analysis`

| Export | Kind | Signature |
|---|---|---|
| `LieInvariantCoordinates` | class | `(invariants: 'tuple[sp.Expr, ...]', transverse: 'tuple[sp.Expr, ...]', jacobian: 'sp.Expr', validity_conditions: 'tuple[sp.Expr, ...]') -> None` |
| `LiePointSymmetryAnalysis` | class | `(determining_equations: 'tuple[sp.Equality, ...]', xi_functions: 'tuple[sp.Expr, ...]', phi_function: 'sp.Expr', polynomial_solution: 'object \| None', generators: 'tuple[tuple[tuple[sp.Expr, ...], sp.Expr], ...]') -> None` |
| `analyze_lie_point_symmetries` | function | `(equation, dep_function, indep_vars, *, polynomial_degree: 'int' = 1, include_dependent_var: 'bool' = True)` |
| `invariants_from_point_generator` | function | `(indep_vars, dep_symbol, xi_coeffs, phi=0)` |

## `pdesolve.method_names`

| Export | Kind | Signature |
|---|---|---|
| `normalize_method_name` | function | `(name: 'str') -> 'str'` |

## `pdesolve.planners.coordinator`

| Export | Kind | Signature |
|---|---|---|
| `plan_canonical_problem` | function | `(problem, **preferences)` |

## `pdesolve.problem`

| Export | Kind | Signature |
|---|---|---|
| `PDEProblem` | class | `(equation: 'sp.Equality', dep_function: 'sp.Expr', indep_vars: 'tuple[sp.Symbol, ...]', ics: 'Any' = None, bcs: 'Any' = None, domain: 'Any' = None, assumptions: 'Any' = True, profile: 'Any' = None, normalized_data: 'Any' = None, details: 'dict[str, Any]' = <factory>, canonical_representation: 'CanonicalPDERepresentation \| None' = None) -> None` |
| `build_pde_problem` | function | `(eq_or_expr, dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None, domain=None, assumptions=True)` |

## `pdesolve.recognition`

| Export | Kind | Signature |
|---|---|---|
| `PDERecognitionRecord` | class | `(family: 'str', recognized: 'bool', solver_hint: 'str \| None' = None, tags: 'tuple[str, ...]' = (), metadata: 'dict[str, Any]' = <factory>) -> None` |
| `build_canonical_representation` | function | `(profile: 'PDEProblemProfile', *, ics=None, bcs=None, normalized_data=None, dep_expr=None, domain=None) -> 'CanonicalPDERepresentation'` |
| `recognize_pde_structure` | function | `(profile: 'PDEProblemProfile', *, canonical: 'CanonicalPDERepresentation \| None' = None) -> 'tuple[PDERecognitionRecord, ...]'` |

## `pdesolve.recognizers.coordinator`

| Export | Kind | Signature |
|---|---|---|
| `recognize_canonical_problem` | function | `(problem)` |

## `pdesolve.result_verification`

| Export | Kind | Signature |
|---|---|---|
| `ClassicalResidualVerifier` | class | `(name: 'str' = 'classical_residual') -> None` |
| `ImplicitSolutionVerifier` | class | `(name: 'str' = 'implicit_relation') -> None` |
| `KernelVerifier` | class | `(name: 'str' = 'kernel') -> None` |
| `SeriesVerifier` | class | `(name: 'str' = 'series') -> None` |
| `TransformVerifier` | class | `(name: 'str' = 'transform_structural') -> None` |
| `VerificationStrategy` | class | `(*args, **kwargs)` |
| `WeakSolutionVerifier` | class | `(name: 'str' = 'weak_solution') -> None` |
| `select_verification_strategy` | function | `(result: 'Any') -> 'VerificationStrategy'` |
| `verify_result` | function | `(equation, result, dep_function, indep_vars, *, ics=None, bcs=None, assumptions=True) -> 'PDEVerificationSummary'` |

## `pdesolve.results`

| Export | Kind | Signature |
|---|---|---|
| `BasePDEResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = ()) -> None` |
| `CanonicalPDERepresentation` | class | `(dependent_variables: 'tuple[sp.Expr, ...]', independent_variables: 'tuple[sp.Symbol, ...]', normalized_equations: 'tuple[sp.Equality, ...]', order: 'int', linearity: 'str', principal_part: 'Any' = None, coefficient_dependence: 'tuple[str, ...]' = (), principal_multiindex: 'Any' = None, ic_metadata: 'dict[str, Any]' = <factory>, bc_metadata: 'dict[str, Any]' = <factory>, domain_metadata: 'dict[str, Any]' = <factory>, geometry_metadata: 'dict[str, Any]' = <factory>, time_slice_metadata: 'dict[str, Any]' = <factory>, weak_solution_flags: 'tuple[str, ...]' = (), transformability_tags: 'tuple[str, ...]' = (), recognized_tags: 'tuple[str, ...]' = (), details: 'dict[str, Any]' = <factory>) -> None` |
| `ClosedFormPDEResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = ()) -> None` |
| `ConservationLawCanonicalForm` | class | `(indep_vars: 'tuple[sp.Symbol, sp.Symbol]', dep_function: 'sp.Expr', density: 'sp.Expr', flux: 'sp.Expr', source: 'sp.Expr' = 0, normalized_equation: 'sp.Equality \| None' = None, autonomous_flux: 'sp.Expr \| None' = None, family: 'str' = 'scalar_conservation_law', details: 'dict[str, Any]' = <factory>) -> None` |
| `ConservationLawImplicitCharacteristicResult` | class | `(method: 'str', solution: 'sp.Equality \| tuple[sp.Equality, ...]', profile: 'Any', characteristic_parameter: 'Any', characteristic_relation: 'sp.Equality', profile_relation: 'sp.Equality', footpoint_equation: 'sp.Equality \| None' = None, implicit_relation: 'sp.Equality \| None' = None, characteristic_speed: 'Any' = None, canonical_form: 'ConservationLawCanonicalForm \| None' = None, initial_data: 'ConservationLawInitialData1D \| None' = None, details: 'dict[str, Any]' = <factory>) -> None` |
| `ConservationLawInitialData1D` | class | `(indep_vars: 'tuple[sp.Symbol, sp.Symbol]', dep_function: 'sp.Expr', kind: 'str', profile: 'Any' = None, equation: 'sp.Equality \| None' = None, left_state: 'Any' = None, right_state: 'Any' = None, interface: 'Any' = None, details: 'dict[str, Any]' = <factory>) -> None` |
| `ConservationLawPropagationResult` | class | `(method: 'str', solution: 'sp.Equality', profile: 'Any', speed: 'Any', canonical_form: 'ConservationLawCanonicalForm \| None' = None, initial_data: 'ConservationLawInitialData1D \| None' = None, details: 'dict[str, Any]' = <factory>) -> None` |
| `ConservationLawRarefactionResult` | class | `(method: 'str', solution: 'sp.Equality', flux: 'sp.Expr', left_state: 'Any', right_state: 'Any', left_speed: 'Any', right_speed: 'Any', self_similar_variable: 'Any', canonical_form: 'ConservationLawCanonicalForm \| None' = None, initial_data: 'ConservationLawInitialData1D \| None' = None, details: 'dict[str, Any]' = <factory>) -> None` |
| `ConservationLawShockResult` | class | `(method: 'str', solution: 'sp.Equality', flux: 'sp.Expr', left_state: 'Any', right_state: 'Any', shock_speed: 'Any', canonical_form: 'ConservationLawCanonicalForm \| None' = None, initial_data: 'ConservationLawInitialData1D \| None' = None, details: 'dict[str, Any]' = <factory>) -> None` |
| `EigenfunctionExpansionResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), series_terms: 'Any' = None, eigen_data: 'Any' = None) -> None` |
| `FundamentalSolutionResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), kernel: 'Any' = None, source_point: 'Any' = None, operator_family: 'str \| None' = None) -> None` |
| `GreenFunctionResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), kernel: 'Any' = None, source_point: 'Any' = None, operator_family: 'str \| None' = None, boundary_type: 'str \| None' = None) -> None` |
| `ImplicitPDEResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = ()) -> None` |
| `KernelRepresentationResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), kernel: 'Any' = None, source_point: 'Any' = None, operator_family: 'str \| None' = None) -> None` |
| `NumericalFallbackResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), numerical_object: 'Any' = None) -> None` |
| `PDEExecutionTrace` | class | `(selected_method: 'str \| None', attempted_methods: 'tuple[str, ...]', steps: 'tuple[PDESolverTraceStep, ...]' = ()) -> None` |
| `PDEProblemProfile` | class | `(indep_vars: 'tuple[sp.Symbol, ...]', dep_function: 'sp.Expr', normalized_equation: 'sp.Equality', zero_expression: 'sp.Expr', order: 'int', principal_solved_form: 'object \| None', characteristic_data: 'object \| None', first_order_linear: 'object \| None', second_order_class: 'object \| None', canonical_family: 'str \| None', conservation_law: 'object \| None', details: 'dict[str, Any]' = <factory>) -> None` |
| `PDESolutionPlan` | class | `(profile: 'PDEProblemProfile', steps: 'tuple[PDESolverMethodCandidate, ...]', details: 'dict[str, Any]' = <factory>) -> None` |
| `PDESolutionRecord` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), steps: 'tuple[str, ...]' = (), canonical_equation: 'sp.Equality \| None' = None) -> None` |
| `PDESolverMethodCandidate` | class | `(method: 'str', score: 'int', reasons: 'tuple[str, ...]', details: 'dict[str, Any]' = <factory>) -> None` |
| `PDESolverTraceStep` | class | `(method: 'str', stage: 'str', success: 'bool', message: 'str' = '', exception_type: 'str \| None' = None, verification: 'PDEVerificationSummary \| None' = None) -> None` |
| `PDEVerificationSummary` | class | `(verified: 'bool \| None', status: 'str', pde_verified: 'bool \| None' = None, initial_verified: 'bool \| None' = None, boundary_verified: 'bool \| None' = None, pde_residual: 'Any' = None, initial_residuals: 'tuple[Any, ...]' = (), boundary_residuals: 'tuple[Any, ...]' = (), mode: 'str' = 'unknown', message: 'str' = '') -> None` |
| `SeriesPDEResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), series_terms: 'Any' = None) -> None` |
| `SolverMethodResult` | class | `(*, method_family: 'str', solution: 'Any', details: 'dict[str, Any]' = <factory>) -> None` |
| `SystemPDEResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), system_size: 'int' = 1, transform: 'Any' = None, characteristic_variables: 'Any' = None) -> None` |
| `TransformPDEResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), transform_data: 'Any' = None) -> None` |
| `UnsolvedButReducedResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = ()) -> None` |
| `WeakSolutionResult` | class | `(method: 'str', solution: 'Any', classification: 'Any' = None, assumptions: 'Any' = True, verification: 'dict[str, Any]' = <factory>, metadata: 'dict[str, Any]' = <factory>, reduced_problem: 'Any' = None, warnings: 'tuple[str, ...]' = (), admissibility: 'dict[str, Any]' = <factory>) -> None` |

## `pdesolve.separation_framework`

| Export | Kind | Signature |
|---|---|---|
| `SeparableGeometryPlan` | class | `(geometry_kind: 'str', boundary_family: 'str', eigenbasis: 'str', spatial_variables: 'tuple[sp.Symbol, ...]', metadata: 'dict[str, Any]' = <factory>) -> None` |
| `build_separable_geometry_plan` | function | `(geometry: 'DomainGeometry \| None', conditions: 'ConditionModel \| None', *, family: 'str \| None' = None) -> 'SeparableGeometryPlan \| None'` |

## `pdesolve.separation_general`

| Export | Kind | Signature |
|---|---|---|
| `ProductSeparationResult` | class | `(ansatz: 'sp.Equality', separated_expression: 'sp.Expr', separation_constant: 'sp.Symbol', factor_equations: 'tuple[sp.Equality, ...]', factors: 'tuple[sp.Expr, ...]', variables: 'tuple[sp.Symbol, ...]', verified_separable: 'bool') -> None` |
| `separate_product_pde` | function | `(equation, dep_function, indep_vars, *, separation_constant=None) -> 'ProductSeparationResult'` |

## `pdesolve.solver_execution`

| Export | Kind | Signature |
|---|---|---|
| `solve_with_canonical_problem` | function | `(problem, method: 'str', **kwargs) -> 'BasePDEResult'` |

## `pdesolve.solvers.coordinator`

| Export | Kind | Signature |
|---|---|---|
| `execute_planned_solver` | function | `(problem, method, **kwargs)` |

## `pdesolve.special_pdes`

| Export | Kind | Signature |
|---|---|---|
| `SpecialPDEResult` | class | `(solution_family: 'sp.Expr', family_name: 'str') -> None` |
| `recognize_heat_or_advection_diffusion` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'Optional[SpecialPDEResult]'` |
| `recognize_laplace_or_helmholtz` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'Optional[SpecialPDEResult]'` |
| `solve_special_pde` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'Optional[SpecialPDEResult]'` |

## `pdesolve.sturm_liouville`

| Export | Kind | Signature |
|---|---|---|
| `SturmLiouvilleProblem` | class | `(variable: 'sp.Symbol', function: 'sp.Expr', p: 'sp.Expr', q: 'sp.Expr', weight: 'sp.Expr', interval: 'tuple[sp.Expr, sp.Expr]', left_boundary: 'str' = 'dirichlet', right_boundary: 'str' = 'dirichlet') -> None` |
| `SturmLiouvilleSpectrum` | class | `(problem: 'SturmLiouvilleProblem', index: 'sp.Symbol', eigenvalues: 'sp.Expr', eigenfunctions: 'sp.Expr', norm_squared: 'sp.Expr', orthogonality_weight: 'sp.Expr', includes_zero_mode: 'bool' = False) -> None` |
| `solve_regular_constant_sturm_liouville` | function | `(problem: 'SturmLiouvilleProblem') -> 'SturmLiouvilleSpectrum'` |

## `pdesolve.transform_framework`

| Export | Kind | Signature |
|---|---|---|
| `TransformMethodPlan` | class | `(method: 'str', domain: 'str', transform_family: 'str', required_conditions: 'tuple[str, ...]' = (), metadata: 'dict[str, Any]' = <factory>) -> None` |
| `build_transform_method_plan` | function | `(canonical: 'CanonicalPDERepresentation \| None', geometry: 'DomainGeometry \| None', conditions: 'ConditionModel \| None') -> 'TransformMethodPlan \| None'` |

## `pdesolve.transform_postprocess`

| Export | Kind | Signature |
|---|---|---|
| `TransformPostprocessReport` | class | `(changed: 'bool', stages: 'tuple[str, ...]', skipped_reason: 'str \| None' = None) -> None` |
| `evaluate_inner_transforms` | function | `(expr, *, max_ops: 'int' = 60)` |
| `postprocess_transform_result` | function | `(result: 'Any', *, max_ops: 'int' = 60)` |

## `pdesolve.unified_transform`

| Export | Kind | Signature |
|---|---|---|
| `EvolutionPDEProfile` | class | `(family_name: 'str', order_x: 'int', time_coefficient: 'sp.Expr', coeffs: 'dict[int, sp.Expr]', dispersion_relation: 'sp.Expr', notes: 'str' = '') -> None` |
| `UnifiedTransformResult` | class | `(*, method_family: 'str', solution: 'Any', details: 'dict[str, Any]' = <factory>, dispersion_relation: 'sp.Expr', domain: 'str', is_formal: 'bool' = False, notes: 'str' = '', profile: 'Optional[EvolutionPDEProfile]' = None) -> None` |
| `determine_dispersion_relation` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'sp.Expr'` |
| `recognize_evolution_pde` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'Optional[EvolutionPDEProfile]'` |
| `solve_unified_transform` | function | `(eq: 'sp.Equality \| sp.Expr', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]', *, initial_condition: 'Optional[sp.Equality]' = None, boundary_conditions: 'Optional[Sequence[sp.Equality]]' = None, domain: 'str' = 'whole_line') -> 'UnifiedTransformResult'` |
| `solve_unified_transform_half_line` | function | `(eq: 'sp.Equality \| sp.Expr', initial_condition: 'sp.Equality', boundary_conditions: 'Sequence[sp.Equality]', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'UnifiedTransformResult'` |
| `solve_unified_transform_whole_line` | function | `(eq: 'sp.Equality \| sp.Expr', initial_condition: 'sp.Equality', u: 'sp.Function', vars: 'tuple[sp.Symbol, sp.Symbol]') -> 'UnifiedTransformResult'` |

