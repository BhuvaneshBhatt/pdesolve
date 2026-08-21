# Methods and solver inventory

PDESolve distinguishes public solving functions, planner method keys, and internal helper functions. The canonical method keys below are the values accepted by the execution registry.

## Canonical execution methods (35)

| Method key | Solver family |
|---|---|
| `burgers_implicit` | `burgers` |
| `charpit` | `complete_integral` |
| `classification_only` | `classification` |
| `complete_integral` | `complete_integral` |
| `conservation_law` | `conservation_law` |
| `constant_coefficient_inverse_operator` | `constant_coefficient` |
| `first_order` | `first_order_linear` |
| `first_order_nonlinear_auto` | `first_order_nonlinear` |
| `fourier_heat` | `transform_heat` |
| `generalized_clairaut_complete_integral` | `complete_integral` |
| `heat_dirichlet_series` | `series_heat` |
| `heat_half_line_transform` | `transform_heat` |
| `heat_laplace_transform` | `transform_heat` |
| `heat_neumann_series` | `series_heat` |
| `heat_robin_series` | `series_heat` |
| `heat_whole_line` | `heat` |
| `hyperbolic_system` | `system` |
| `invariant_reduction_auto` | `invariant_reduction` |
| `jacobi` | `complete_integral` |
| `kernel_fundamental_solution` | `kernel` |
| `kernel_green_function` | `kernel` |
| `laplace_fourier_heat` | `transform_heat` |
| `laplace_rectangle_dirichlet_series` | `elliptic_series` |
| `post_reduction_auto` | `reduction` |
| `quasilinear_implicit` | `first_order_quasilinear` |
| `separation_framework` | `separation` |
| `separation_of_variables` | `separation` |
| `structured_transform` | `transform_framework` |
| `symmetry_reduction` | `symmetry` |
| `transport_ivp` | `transport` |
| `unified_transform` | `unified_transform` |
| `wave_dalembert` | `wave` |
| `wave_dirichlet_series` | `series_wave` |
| `wave_laplace_sine_transform` | `transform_wave` |
| `wave_laplace_transform` | `transform_wave` |

## Recognition and planning

Recognition is layered: canonicalization records equation structure; family recognizers identify applicable mathematical forms; condition and domain analysis add geometry/data constraints; the planner ranks executable method keys; the solver coordinator executes the selected method.

Important recognition and planning entry points include `build_canonical_representation(...)`, `recognize_pde_structure(...)`, `recognize_canonical_problem(...)`, `plan_canonical_problem(...)`, `build_separable_geometry_plan(...)`, `build_transform_method_plan(...)`, and `build_kernel_method_plan(...)`.

## Focused solver APIs

Focused functions remain useful when the formulation itself matters, including first-order linear/nonlinear solvers, complete-integral and Cauchy solvers, conservation-law solvers, unified transforms, hyperbolic systems, Sturm–Liouville analysis, and Green/fundamental-solution APIs.

## Formal representations

Some transform and reduction functions intentionally return unevaluated integral, series, or implicit representations. These are valid symbolic endpoints and are verified according to their representation rather than being forced into a closed form.
