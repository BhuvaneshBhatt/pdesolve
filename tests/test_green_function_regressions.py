import sympy as sp

from pdesolve import solve_green_function
from pdesolve.domains import (
    RectangleDomain,
    DomainGeometry,
    HalfPlaneDomain,
    HalfSpaceDomain,
)


def test_laplace_strip_dirichlet_green_function():
    x, y, a = sp.symbols("x y a", positive=True)
    u = sp.Function("u")
    geom = RectangleDomain("strip", (x, y), {"x": (-sp.oo, sp.oo), "y": (0, a)})
    eq = sp.Eq(
        sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2),
        sp.DiracDelta(x) * sp.DiracDelta(y - a / 2),
    )
    res = solve_green_function(
        eq, u(x, y), (x, y), bcs=["DirichletCondition"], geometry=geom
    )
    assert res.solution.has(sp.log) or res.solution.has(sp.cosh)
    assert res.metadata["geometry_kind"] == "strip"


def test_laplace_semi_infinite_strip_dirichlet_green_function():
    x, y, a = sp.symbols("x y a", positive=True)
    u = sp.Function("u")
    geom = RectangleDomain(
        "semi_infinite_strip", (x, y), {"x": (0, sp.oo), "y": (0, a)}
    )
    eq = sp.Eq(
        sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2),
        sp.DiracDelta(x - 1) * sp.DiracDelta(y - a / 2),
    )
    res = solve_green_function(
        eq, u(x, y), (x, y), bcs=["DirichletCondition"], geometry=geom
    )
    assert res.solution.has(sp.log) or res.solution.has(sp.cosh)
    assert res.metadata["geometry_kind"] == "semi_infinite_strip"


def test_helmholtz_quadrant_neumann_green_function():
    x, y, kappa = sp.symbols("x y kappa", positive=True)
    u = sp.Function("u")
    geom = DomainGeometry("quadrant", (x, y), {"x": (0, sp.oo), "y": (0, sp.oo)})
    eq = sp.Eq(
        sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2) + kappa * u(x, y),
        sp.DiracDelta(x - 1) * sp.DiracDelta(y - 2),
    )
    res = solve_green_function(eq, u(x, y), (x, y), bcs=["NeumannValue"], geometry=geom)
    assert res.solution.has(sp.hankel2)
    assert res.boundary_type == "neumann"


def test_helmholtz_strip_dirichlet_green_function_series():
    x, y, a, kappa = sp.symbols("x y a kappa", positive=True)
    u = sp.Function("u")
    geom = RectangleDomain("strip", (x, y), {"x": (-sp.oo, sp.oo), "y": (0, a)})
    eq = sp.Eq(
        sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2) + kappa * u(x, y),
        sp.DiracDelta(x - 1) * sp.DiracDelta(y - a / 3),
    )
    res = solve_green_function(
        eq, u(x, y), (x, y), bcs=["DirichletCondition"], geometry=geom
    )
    assert res.solution.has(sp.Sum)


def test_heat_half_plane_dirichlet_image_kernel():
    x, y, t = sp.symbols("x y t", real=True)
    u = sp.Function("u")
    geom = HalfPlaneDomain("half_plane", (x, y), {"y": (0, sp.oo)})
    eq = sp.Eq(
        sp.diff(u(x, y, t), t) - sp.diff(u(x, y, t), x, 2) - sp.diff(u(x, y, t), y, 2),
        sp.DiracDelta(x) * sp.DiracDelta(y - 1) * sp.DiracDelta(t),
    )
    res = solve_green_function(
        eq, u(x, y, t), (x, y, t), bcs=["DirichletCondition"], geometry=geom
    )
    assert res.solution.has(sp.exp)
    assert res.metadata["verification"].get("distributional_plausibility") is True


def test_wave_half_space_neumann_image_kernel():
    x, y, z, t = sp.symbols("x y z t", real=True)
    u = sp.Function("u")
    geom = HalfSpaceDomain("half_space", (x, y, z), {"z": (0, sp.oo)})
    eq = sp.Eq(
        sp.diff(u(x, y, z, t), t, 2)
        - sp.diff(u(x, y, z, t), x, 2)
        - sp.diff(u(x, y, z, t), y, 2)
        - sp.diff(u(x, y, z, t), z, 2),
        sp.DiracDelta(x) * sp.DiracDelta(y) * sp.DiracDelta(z - 1) * sp.DiracDelta(t),
    )
    res = solve_green_function(
        eq, u(x, y, z, t), (x, y, z, t), bcs=["NeumannValue"], geometry=geom
    )
    assert res.solution.has(sp.DiracDelta)
    assert res.metadata["geometry_kind"] == "half_space"


def test_half_space_laplace_green_has_boundary_verification_metadata():
    x, y, z = sp.symbols("x y z", real=True)
    u = sp.Function("u")
    geom = HalfSpaceDomain("half_space", (x, y, z), {"z": (0, sp.oo)})
    eq = sp.Eq(
        sp.diff(u(x, y, z), x, 2)
        + sp.diff(u(x, y, z), y, 2)
        + sp.diff(u(x, y, z), z, 2),
        sp.DiracDelta(x) * sp.DiracDelta(y) * sp.DiracDelta(z - 1),
    )
    res = solve_green_function(
        eq, u(x, y, z), (x, y, z), bcs=["DirichletCondition"], geometry=geom
    )
    assert "verification" in res.metadata
