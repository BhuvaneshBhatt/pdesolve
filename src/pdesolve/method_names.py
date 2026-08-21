"""Canonical method-name helpers used by the dispatcher and planner."""

from __future__ import annotations

_CANON = {
    "first_integrals": "first_order",
    "first_integral": "first_order",
    "charpit_method": "charpit",
    "complete_integrals": "complete_integral",
    "complete_integral_method": "complete_integral",
    "inv_reduction": "invariant_reduction_auto",
    "symmetry": "symmetry_reduction",
    "heat_half_line_dirichlet": "heat_half_line_transform",
    "heat_half_line_neumann": "heat_half_line_transform",
}


def normalize_method_name(name: str) -> str:
    return _CANON.get(name, name)
