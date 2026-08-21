from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import sympy as sp

import pdesolve.solver_execution as solver_execution
from pdesolve import classical as classical_mod
from pdesolve import classical_methods as classical_core
from pdesolve.constant_coeff import pdesolve_constant_coefficient
from pdesolve.first_order_linear import solve_first_order_linear_pde
from pdesolve.first_order_nonlinear import solve_first_order_nonlinear_auto
from pdesolve.hyperbolic_system import solve_hyperbolic_system
from pdesolve.kernels import solve_fundamental_solution, solve_green_function
from pdesolve.problem import build_pde_problem
from pdesolve.results import BasePDEResult
from pdesolve.solver_execution import SolverExecutionContext, solve_with_canonical_problem
from pdesolve.unified_transform import solve_unified_transform


@dataclass(frozen=True)
class MethodCoverageSpec:
    family: str
    probe: Callable[[], object]


def _xy():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    return x, y, u


def _xt(*, positive: bool = False):
    x, t = sp.symbols("x t", real=True, positive=positive)
    u = sp.Function("u")
    return x, t, u


def _probe_first_order_linear():
    x, y, u = _xy()
    return solve_first_order_linear_pde(
        sp.Eq(sp.diff(u(x, y), x) + sp.diff(u(x, y), y), 0),
        u,
        (x, y),
    )


def _probe_first_order_nonlinear():
    x, t, u = _xt()
    return solve_first_order_nonlinear_auto(
        sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0),
        u(x, t),
        (x, t),
        ics={"initial_profile": 1 / (x + 1), "curve_value": 0},
    )


def _probe_classification_only():
    x, t, u = _xt()
    problem = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0),
        u(x, t),
        (x, t),
    )
    return solve_with_canonical_problem(problem, "classification_only")


def _probe_transport():
    x, t, u = _xt()
    return classical_mod.solve_transport_ivp(
        sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0),
        u(x, t),
        (x, t),
        initial_profile=sp.exp(-(x**2)),
        initial_curve_value=0,
    )


def _probe_quasilinear():
    x, t, u = _xt()
    return classical_mod.solve_quasilinear_pde_characteristics_implicit(
        sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0),
        u(x, t),
        (x, t),
        initial_profile=1 / (x + 1),
        initial_curve_value=0,
    )


def _probe_conservation():
    x, t, u = _xt()
    problem = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0),
        u(x, t),
        (x, t),
    )
    return solve_with_canonical_problem(problem, "conservation_law")


def _probe_burgers():
    x, t, u = _xt()
    return classical_mod.solve_inviscid_burgers_ivp_implicit(
        u(x, t), x=x, t=t, initial_profile=sp.sin(x)
    )


def _probe_wave_dalembert():
    x, t, u = _xt()
    return classical_mod.solve_wave_equation_1d_ivp(
        u(x, t),
        x=x,
        t=t,
        wave_speed=1,
        initial_displacement=sp.sin(x),
        initial_velocity=0,
    )


def _probe_heat_whole_line():
    x, t, u = _xt()
    return classical_mod.solve_heat_equation_1d_whole_line_ivp(
        u(x, t), x=x, t=t, diffusivity=1, initial_profile=sp.exp(-(x**2))
    )


def _probe_fourier_heat():
    x, t, u = _xt()
    return classical_mod.solve_heat_equation_1d_fourier_transform(
        u(x, t), x=x, t=t, diffusivity=1, initial_profile=sp.exp(-(x**2))
    )


def _probe_heat_dirichlet():
    x, t, u = _xt(positive=True)
    return classical_mod.solve_heat_equation_1d_dirichlet_series(
        u(x, t), x=x, t=t, initial_profile=x * (sp.pi - x), terms=3
    )


def _probe_heat_neumann():
    x, t, u = _xt(positive=True)
    return classical_mod.solve_heat_equation_1d_neumann_series(
        u(x, t), x=x, t=t, initial_profile=1 + sp.cos(x), terms=3
    )


def _probe_heat_robin():
    x, t, u = _xt(positive=True)
    return classical_mod.solve_heat_equation_1d_robin_series(
        u(x, t), x=x, t=t, initial_profile=x * (sp.pi - x), terms=3
    )


def _probe_wave_dirichlet():
    x, t, u = _xt(positive=True)
    return classical_mod.solve_wave_equation_1d_dirichlet_series(
        u(x, t),
        x=x,
        t=t,
        initial_displacement=sp.sin(x),
        initial_velocity=0,
        terms=3,
    )


def _probe_separation():
    x, t, u = _xt(positive=True)
    return classical_mod.separate_variables_structured(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        bcs={"type": "dirichlet_homogeneous_interval", "length": sp.pi},
    )


def _probe_rectangle():
    x, y, u = _xy()
    return classical_mod.solve_rectangle_dirichlet_laplace_series(
        u(x, y), x=x, y=y, boundary_top=x * (sp.pi - x), terms=3
    )


def _probe_heat_half_line():
    x, t, u = _xt(positive=True)
    return classical_mod.solve_heat_equation_1d_half_line_transform(
        u(x, t), x=x, t=t, initial_profile=sp.Integer(0), boundary="dirichlet"
    )


def _probe_heat_laplace():
    x, t, u = _xt(positive=True)
    return classical_mod.solve_heat_equation_1d_laplace_transform_formal(
        u(x, t), x=x, t=t, initial_profile=sp.sin(x)
    )


def _probe_laplace_fourier_heat():
    x, t, u = _xt(positive=True)
    return classical_mod.solve_heat_equation_1d_laplace_fourier_formal(
        u(x, t), x=x, t=t, initial_profile=sp.exp(-(x**2))
    )


def _probe_wave_laplace():
    x, t, u = _xt(positive=True)
    return classical_mod.solve_wave_equation_1d_laplace_transform_formal(
        u(x, t), x=x, t=t, initial_displacement=sp.sin(x), initial_velocity=0
    )


def _probe_wave_laplace_sine():
    x, t, u = _xt(positive=True)
    return classical_mod.solve_wave_equation_1d_laplace_sine_transform_formal(
        u(x, t), x=x, t=t, initial_displacement=sp.sin(x), initial_velocity=0
    )


def _probe_structured_transform():
    x, t, u = _xt()
    problem = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics={"initial_profile": sp.exp(-(x**2))},
    )
    return solve_with_canonical_problem(problem, "structured_transform")


def _probe_unified_transform():
    x, t, u = _xt(positive=True)
    eq = sp.Eq(sp.I * sp.diff(u(x, t), t) + sp.diff(u(x, t), x, 2), 0)
    ic = sp.Eq(u(x, 0), sp.exp(-x))
    return solve_unified_transform(eq, u, (x, t), initial_condition=ic, domain="whole_line")


def _probe_symmetry():
    x, t, u = _xt()
    return classical_core._solve_via_symmetry_workflow(
        sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0),
        u(x, t),
        (x, t),
        max_symmetry_steps=1,
    )


def _probe_post_reduction():
    z = sp.symbols("z", real=True)
    f = sp.Function("f")
    return classical_mod.solve_reduced_equation_auto(sp.Eq(sp.diff(f(z), z), f(z)))


def _probe_generalized_clairaut():
    x, y, u = _xy()
    ux = sp.diff(u(x, y), x)
    uy = sp.diff(u(x, y), y)
    return classical_mod.solve_generalized_clairaut_complete_integral(
        sp.Eq(u(x, y), x * ux + y * uy + ux**2 + uy**2), u(x, y), (x, y)
    )


def _complete_integral_equation():
    x, y, u = _xy()
    return x, y, u, sp.Eq(sp.diff(u(x, y), x) * sp.diff(u(x, y), y), 1)


def _probe_charpit():
    x, y, u, eq = _complete_integral_equation()
    return classical_mod.solve_charpit_complete_integral_2vars(eq, u(x, y), (x, y))


def _probe_complete_integral():
    x, y, u, eq = _complete_integral_equation()
    return classical_mod.solve_complete_integral_pde(eq, u(x, y), (x, y))


def _probe_jacobi():
    x, y, z = sp.symbols("x y z", real=True)
    u = sp.Function("u")
    dep = u(x, y, z)
    eq = sp.Eq(sp.diff(dep, x) ** 2 + sp.diff(dep, y) ** 2 + sp.diff(dep, z) ** 2, 1)
    return classical_mod.solve_jacobi_complete_integral(eq, dep, (x, y, z))


def _probe_hyperbolic_system():
    x, t = sp.symbols("x t", real=True)
    u1, u2 = sp.Function("u1"), sp.Function("u2")
    eqs = [
        sp.Eq(sp.diff(u1(t, x), t), sp.diff(u2(t, x), x)),
        sp.Eq(sp.diff(u2(t, x), t), sp.diff(u1(t, x), x)),
    ]
    ics = [sp.Eq(u1(0, x), sp.sin(x)), sp.Eq(u2(0, x), sp.cos(x))]
    return solve_hyperbolic_system(eqs, ics, [u1, u2], (x, t))


def _probe_constant_coeff():
    x, t, u = _xt()
    return pdesolve_constant_coefficient(
        sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x), 0),
        u(x, t),
        (x, t),
    )


def _probe_kernel_fundamental():
    x, t, u = _xt()
    return solve_fundamental_solution(
        sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0),
        u(x, t),
        (x, t),
    )


def _probe_kernel_green():
    x, t, u = _xt()
    return solve_green_function(
        sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0),
        u(x, t),
        (x, t),
        geometry="half_line",
        bcs={"type": "dirichlet"},
    )


METHOD_COVERAGE_SPECS = {
    "first_order_nonlinear_auto": MethodCoverageSpec(
        "first_order_nonlinear", _probe_first_order_nonlinear
    ),
    "invariant_reduction_auto": MethodCoverageSpec("invariant_reduction", _probe_symmetry),
    "classification_only": MethodCoverageSpec("classification", _probe_classification_only),
    "first_order": MethodCoverageSpec("first_order_linear", _probe_first_order_linear),
    "transport_ivp": MethodCoverageSpec("transport", _probe_transport),
    "quasilinear_implicit": MethodCoverageSpec("first_order_quasilinear", _probe_quasilinear),
    "conservation_law": MethodCoverageSpec("conservation_law", _probe_conservation),
    "burgers_implicit": MethodCoverageSpec("burgers", _probe_burgers),
    "wave_dalembert": MethodCoverageSpec("wave", _probe_wave_dalembert),
    "heat_whole_line": MethodCoverageSpec("heat", _probe_heat_whole_line),
    "fourier_heat": MethodCoverageSpec("transform_heat", _probe_fourier_heat),
    "heat_dirichlet_series": MethodCoverageSpec("series_heat", _probe_heat_dirichlet),
    "heat_neumann_series": MethodCoverageSpec("series_heat", _probe_heat_neumann),
    "heat_robin_series": MethodCoverageSpec("series_heat", _probe_heat_robin),
    "wave_dirichlet_series": MethodCoverageSpec("series_wave", _probe_wave_dirichlet),
    "separation_framework": MethodCoverageSpec("separation", _probe_heat_dirichlet),
    "separation_of_variables": MethodCoverageSpec("separation", _probe_separation),
    "laplace_rectangle_dirichlet_series": MethodCoverageSpec("elliptic_series", _probe_rectangle),
    "heat_half_line_transform": MethodCoverageSpec("transform_heat", _probe_heat_half_line),
    "heat_laplace_transform": MethodCoverageSpec("transform_heat", _probe_heat_laplace),
    "laplace_fourier_heat": MethodCoverageSpec("transform_heat", _probe_laplace_fourier_heat),
    "wave_laplace_transform": MethodCoverageSpec("transform_wave", _probe_wave_laplace),
    "wave_laplace_sine_transform": MethodCoverageSpec("transform_wave", _probe_wave_laplace_sine),
    "structured_transform": MethodCoverageSpec("transform_framework", _probe_structured_transform),
    "unified_transform": MethodCoverageSpec("unified_transform", _probe_unified_transform),
    "symmetry_reduction": MethodCoverageSpec("symmetry", _probe_symmetry),
    "post_reduction_auto": MethodCoverageSpec("reduction", _probe_post_reduction),
    "generalized_clairaut_complete_integral": MethodCoverageSpec(
        "complete_integral", _probe_generalized_clairaut
    ),
    "charpit": MethodCoverageSpec("complete_integral", _probe_charpit),
    "complete_integral": MethodCoverageSpec("complete_integral", _probe_complete_integral),
    "jacobi": MethodCoverageSpec("complete_integral", _probe_jacobi),
    "hyperbolic_system": MethodCoverageSpec("system", _probe_hyperbolic_system),
    "constant_coefficient_inverse_operator": MethodCoverageSpec(
        "constant_coefficient", _probe_constant_coeff
    ),
    "kernel_fundamental_solution": MethodCoverageSpec("kernel", _probe_kernel_fundamental),
    "kernel_green_function": MethodCoverageSpec("kernel", _probe_kernel_green),
}

CANONICAL_METHODS = tuple(solver_execution._METHOD_REGISTRY)


@pytest.mark.parametrize("method", CANONICAL_METHODS)
def test_every_canonical_method_is_registered_and_documented(method):
    assert method in solver_execution._METHOD_REGISTRY
    assert method in METHOD_COVERAGE_SPECS


@pytest.mark.parametrize("method", CANONICAL_METHODS)
def test_every_canonical_method_dispatches_to_its_registered_handler(method):
    sentinel = object()
    seen = []

    def handler(ctx: SolverExecutionContext):
        seen.append(ctx.method)
        return sentinel

    x, t, u = _xt()
    problem = build_pde_problem(sp.Eq(sp.diff(u(x, t), t), 0), u(x, t), (x, t))
    with solver_execution.temporary_method_handler(method, handler):
        result = solve_with_canonical_problem(problem, method)
    assert seen == [method]
    assert result.metadata["raw_result"] is sentinel


@pytest.mark.parametrize("method", CANONICAL_METHODS)
def test_every_canonical_method_has_representative_mathematical_execution(method):
    result = METHOD_COVERAGE_SPECS[method].probe()
    assert result is not None
    if isinstance(result, BasePDEResult):
        assert result.method
        assert result.solution is not None or result.reduced_problem is not None


@pytest.mark.parametrize("method", CANONICAL_METHODS)
def test_every_canonical_method_has_declared_family_contract(method):
    spec = METHOD_COVERAGE_SPECS[method]
    assert spec.family
    assert callable(spec.probe)


def test_method_coverage_specs_match_registry_exactly():
    assert set(METHOD_COVERAGE_SPECS) == set(CANONICAL_METHODS)
