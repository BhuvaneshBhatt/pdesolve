from __future__ import annotations

from .classical_methods import (
    solve_wave_equation_1d_ivp,
    solve_heat_equation_1d_whole_line_ivp,
    solve_heat_equation_1d_dirichlet_series,
    solve_heat_equation_1d_neumann_series,
    solve_heat_equation_1d_robin_series,
    solve_wave_equation_1d_dirichlet_series,
    solve_laplace_rectangle_dirichlet_series,
)

__all__ = [
    "solve_wave_equation_1d_ivp",
    "solve_heat_equation_1d_whole_line_ivp",
    "solve_heat_equation_1d_dirichlet_series",
    "solve_heat_equation_1d_neumann_series",
    "solve_heat_equation_1d_robin_series",
    "solve_wave_equation_1d_dirichlet_series",
    "solve_laplace_rectangle_dirichlet_series",
]
