from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class BenchmarkCase:
    family: str
    name: str
    equation: Any
    dependent: Any
    variables: tuple[Any, ...]
    ics: Any = None
    bcs: Any = None
    expected_method_hints: tuple[str, ...] = ()
    expected_solution: Any = None
    solution_fragments: tuple[str, ...] = ()
    exact_output_kind: str | None = None
    stress_level: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)


def build_benchmark_cases():
    x, y, t = sp.symbols("x y t", real=True)
    xi = sp.symbols("xi", real=True)
    C1, C2 = sp.symbols("C1 C2", real=True)
    u = sp.Function("u")
    v = sp.Function("v")
    cases = {
        "first_order_linear": (
            BenchmarkCase(
                "first_order_linear",
                "advection",
                sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0),
                u(x, t),
                (x, t),
                expected_method_hints=("first_order", "structured_transform"),
                exact_output_kind="closed_form",
                solution_fragments=("t - x", "x - t", "C[1]", "C1", "F("),
            ),
            BenchmarkCase(
                "first_order_linear",
                "transport_ivp_gaussian",
                sp.Eq(sp.diff(u(x, t), t) + 2 * sp.diff(u(x, t), x), 0),
                u(x, t),
                (x, t),
                ics=sp.Eq(u(x, 0), sp.exp(-(x**2))),
                expected_method_hints=(
                    "transport_ivp",
                    "constant_coefficient_characteristics_ivp",
                    "unified_transform_whole_line",
                ),
                expected_solution=sp.Eq(u(x, t), sp.exp(-((x - 2 * t) ** 2))),
                solution_fragments=("exp",),
                exact_output_kind="closed_form",
            ),
        ),
        "first_order_nonlinear": (
            BenchmarkCase(
                "first_order_nonlinear",
                "clairaut",
                sp.Eq(
                    u(x, y),
                    x * sp.diff(u(x, y), x)
                    + y * sp.diff(u(x, y), y)
                    + sp.sin(sp.diff(u(x, y), x) + sp.diff(u(x, y), y)),
                ),
                u(x, y),
                (x, y),
                expected_method_hints=("generalized_clairaut_complete_integral",),
                expected_solution=sp.Eq(u(x, y), C1 * x + C2 * y + sp.sin(C1 + C2)),
                solution_fragments=("sin",),
                exact_output_kind="complete_integral",
            ),
            BenchmarkCase(
                "first_order_nonlinear",
                "autonomous_charpit",
                sp.Eq(sp.diff(u(x, y), x) ** 2 + sp.diff(u(x, y), y) ** 2, 1),
                u(x, y),
                (x, y),
                expected_method_hints=("charpit", "complete_integral"),
                exact_output_kind="implicit_or_complete_integral",
                stress_level="eikonal",
            ),
            BenchmarkCase(
                "first_order_nonlinear",
                "quasilinear_ivp",
                sp.Eq(sp.diff(u(x, t), x) + u(x, t) * sp.diff(u(x, t), t), 0),
                u(x, t),
                (x, t),
                ics=sp.Eq(u(0, t), 1 / (1 + t)),
                expected_method_hints=(
                    "quasilinear_implicit",
                    "first_order_nonlinear_auto",
                ),
                exact_output_kind="implicit",
                stress_level="characteristics",
            ),
            BenchmarkCase(
                "first_order_nonlinear",
                "nonlinear_complete_integral_stress",
                sp.Eq(sp.diff(u(x, y), x) + sp.diff(u(x, y), y) ** 2, 3),
                u(x, y),
                (x, y),
                expected_method_hints=(
                    "complete_integral",
                    "charpit",
                    "first_order_framework",
                ),
                solution_fragments=("C",),
                exact_output_kind="complete_integral",
                stress_level="symbolic_large",
            ),
        ),
        "conservation_law": (
            BenchmarkCase(
                "conservation_law",
                "burgers_riemann",
                sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0),
                u(x, t),
                (x, t),
                ics=sp.Eq(u(x, 0), sp.Piecewise((1, x < 0), (0, True))),
                expected_method_hints=(
                    "conservation_law",
                    "quasilinear_implicit_characteristics",
                ),
                exact_output_kind="weak_solution",
            ),
            BenchmarkCase(
                "conservation_law",
                "linear_flux_profile",
                sp.Eq(sp.diff(u(x, t), t) + 3 * sp.diff(u(x, t), x), 0),
                u(x, t),
                (x, t),
                ics=sp.Eq(u(x, 0), sp.sin(x)),
                expected_method_hints=(
                    "conservation_law",
                    "transport_ivp",
                    "constant_coefficient_characteristics_ivp",
                    "unified_transform_whole_line",
                ),
                expected_solution=sp.Eq(u(x, t), sp.sin(x - 3 * t)),
                exact_output_kind="closed_form",
            ),
            BenchmarkCase(
                "conservation_law",
                "nonconvex_stress",
                sp.Eq(
                    sp.diff(u(x, t), t)
                    + (u(x, t) ** 3 - u(x, t)) * sp.diff(u(x, t), x),
                    0,
                ),
                u(x, t),
                (x, t),
                ics=sp.Eq(u(x, 0), sp.Piecewise((1, x < 0), (-1, True))),
                expected_method_hints=("conservation_law",),
                exact_output_kind="weak_or_implicit",
                stress_level="nonconvex",
            ),
        ),
        "heat_like": (
            BenchmarkCase(
                "heat_like",
                "whole_line_heat",
                sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
                u(x, t),
                (x, t),
                ics=sp.Eq(u(x, 0), sp.exp(-(x**2))),
                expected_method_hints=(
                    "heat_kernel_whole_line_ivp",
                    "structured_transform",
                    "fourier",
                    "unified_transform_whole_line",
                ),
                expected_solution=sp.Eq(
                    u(x, t),
                    sp.Integral(
                        sp.exp(-(xi**2))
                        * sp.exp(-((x - xi) ** 2) / (4 * t))
                        / (2 * sp.sqrt(sp.pi) * sp.sqrt(t)),
                        (xi, -sp.oo, sp.oo),
                    ),
                ),
                solution_fragments=("sqrt", "exp"),
                exact_output_kind="closed_form",
            ),
            BenchmarkCase(
                "heat_like",
                "whole_line_heat_polynomial_stress",
                sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
                u(x, t),
                (x, t),
                ics=sp.Eq(u(x, 0), x**4 - 6 * x**2 + 3),
                expected_method_hints=("structured_transform", "heat"),
                solution_fragments=("x", "t"),
                exact_output_kind="closed_form_or_series",
                stress_level="symbolic_large",
            ),
            BenchmarkCase(
                "heat_like",
                "interval_heat_dirichlet",
                sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
                u(x, t),
                (x, t),
                ics=sp.Eq(u(x, 0), sp.sin(x)),
                bcs=(sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)),
                expected_method_hints=(
                    "separation_framework",
                    "heat_dirichlet_series",
                    "heat_dirichlet_sine_series",
                    "unified_transform_whole_line",
                ),
                exact_output_kind="series",
            ),
            BenchmarkCase(
                "heat_like",
                "interval_heat_robin",
                sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
                u(x, t),
                (x, t),
                ics=sp.Eq(u(x, 0), x * (sp.pi - x)),
                bcs=(
                    sp.Eq(sp.diff(u(x, t), x).subs(x, 0) + u(0, t), 0),
                    sp.Eq(sp.diff(u(x, t), x).subs(x, sp.pi) + 2 * u(sp.pi, t), 0),
                ),
                expected_method_hints=("separation_framework", "heat_robin_series"),
                exact_output_kind="series",
                stress_level="robin",
            ),
        ),
        "wave_like": (
            BenchmarkCase(
                "wave_like",
                "interval_wave_dirichlet",
                sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
                u(x, t),
                (x, t),
                ics=(
                    sp.Eq(u(x, 0), sp.sin(x)),
                    sp.Eq(sp.diff(u(x, t), t).subs(t, 0), 0),
                ),
                bcs=(sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)),
                expected_method_hints=(
                    "wave_dirichlet_sine_series",
                    "dAlembert_wave_ivp",
                    "separation_framework",
                ),
                expected_solution=sp.Eq(u(x, t), sp.sin(x) * sp.cos(t)),
                exact_output_kind="series",
            ),
            BenchmarkCase(
                "wave_like",
                "whole_line_dalembert",
                sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
                u(x, t),
                (x, t),
                ics=(
                    sp.Eq(u(x, 0), sp.exp(-(x**2))),
                    sp.Eq(sp.diff(u(x, t), t).subs(t, 0), 0),
                ),
                expected_method_hints=("dalembert_wave_ivp", "structured_transform"),
                expected_solution=sp.Eq(
                    u(x, t), sp.exp(-((x - t) ** 2)) / 2 + sp.exp(-((x + t) ** 2)) / 2
                ),
                exact_output_kind="closed_form",
                solution_fragments=("exp",),
            ),
            BenchmarkCase(
                "wave_like",
                "wave_trigonometric_stress",
                sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
                u(x, t),
                (x, t),
                ics=(
                    sp.Eq(u(x, 0), sp.sin(3 * x) + sp.sin(5 * x)),
                    sp.Eq(sp.diff(u(x, t), t).subs(t, 0), 0),
                ),
                bcs=(sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)),
                expected_method_hints=("wave", "series"),
                solution_fragments=("sin", "cos"),
                exact_output_kind="series",
                stress_level="symbolic_large",
            ),
        ),
        "laplace_like": (
            BenchmarkCase(
                "laplace_like",
                "rectangle_laplace",
                sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2), 0),
                u(x, y),
                (x, y),
                bcs=(
                    sp.Eq(u(0, y), 0),
                    sp.Eq(u(sp.pi, y), 0),
                    sp.Eq(u(x, 0), 0),
                    sp.Eq(u(x, sp.pi), sp.sin(x)),
                ),
                expected_method_hints=(
                    "separation_framework",
                    "laplace_rectangle_dirichlet_series",
                ),
                exact_output_kind="series",
                solution_fragments=("sin", "sinh"),
            ),
            BenchmarkCase(
                "laplace_like",
                "rectangle_laplace_polynomial_boundary_stress",
                sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2), 0),
                u(x, y),
                (x, y),
                bcs=(
                    sp.Eq(u(0, y), 0),
                    sp.Eq(u(1, y), 0),
                    sp.Eq(u(x, 0), 0),
                    sp.Eq(u(x, 2), x * (1 - x)),
                ),
                expected_method_hints=("laplace", "series"),
                solution_fragments=("sin", "sinh"),
                exact_output_kind="series",
                stress_level="symbolic_large",
            ),
            BenchmarkCase(
                "laplace_like",
                "disk_harmonic",
                sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2), 0),
                u(x, y),
                (x, y),
                expected_method_hints=("separation_framework",),
                exact_output_kind="series_or_closed_form",
                stress_level="geometry",
                metadata={"geometry": "disk"},
            ),
        ),
        "system_pde": (
            BenchmarkCase(
                "system_pde",
                "hyperbolic_system",
                (
                    sp.Eq(sp.diff(u(x, t), t), sp.diff(v(x, t), x)),
                    sp.Eq(sp.diff(v(x, t), t), sp.diff(u(x, t), x)),
                ),
                (u(x, t), v(x, t)),
                (x, t),
                ics=(sp.Eq(u(x, 0), sp.sin(x)), sp.Eq(v(x, 0), sp.cos(x))),
                expected_method_hints=("hyperbolic_system",),
                expected_solution={
                    u(x, t): sp.sqrt(2) * sp.sin(x) * sp.cos(t + sp.pi / 4),
                    v(x, t): sp.sqrt(2) * sp.sin(t + sp.pi / 4) * sp.cos(x),
                },
                exact_output_kind="closed_form_or_decoupled",
            ),
            BenchmarkCase(
                "system_pde",
                "constant_matrix_advection",
                (
                    sp.Eq(sp.diff(u(x, t), t), sp.diff(v(x, t), x) + u(x, t)),
                    sp.Eq(sp.diff(v(x, t), t), sp.diff(u(x, t), x) - v(x, t)),
                ),
                (u(x, t), v(x, t)),
                (x, t),
                expected_method_hints=("hyperbolic_system", "system_pde"),
                exact_output_kind="formal_or_closed_form",
                stress_level="matrix",
            ),
        ),
    }
    return cases


__all__ = ["BenchmarkCase", "build_benchmark_cases"]
