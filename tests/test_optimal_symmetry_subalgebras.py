import sympy as sp

from pdesolve.geometry import DistributionKD, VectorFieldKD
from pdesolve.symmetry_optimal_systems import commuting_subalgebras, optimal_system_1d


def test_optimal_system_1d_dedupes_scalar_multiples():
    x, y = sp.symbols("x y")
    X = VectorFieldKD((x, y), (1, 0))
    Y = VectorFieldKD((x, y), (2, 0))
    Z = VectorFieldKD((x, y), (0, 1))
    dist = DistributionKD((x, y), (X, Y, Z))

    reps = optimal_system_1d(dist, include_combinations=False)
    sigs = [r.signature for r in reps]

    # one representative for x-translation direction and one for y-translation
    assert len(reps) == 2
    assert any(sig[0] == "translation" and "1" in str(sig) for sig in sigs)


def test_optimal_system_1d_finds_scaling_representative():
    x, y = sp.symbols("x y")
    X = VectorFieldKD((x, y), (x, 0))
    Y = VectorFieldKD((x, y), (2 * x, 0))
    dist = DistributionKD((x, y), (X, Y))

    reps = optimal_system_1d(dist, include_combinations=False)
    assert len(reps) == 1
    assert reps[0].kind == "diagonal_scaling"


def test_commuting_subalgebras_dedupes_obvious_duplicates():
    x, y, t = sp.symbols("x y t")
    X = VectorFieldKD((x, y, t), (1, 0, 0))
    Y = VectorFieldKD((x, y, t), (2, 0, 0))
    Z = VectorFieldKD((x, y, t), (0, 1, 0))
    dist = DistributionKD((x, y, t), (X, Y, Z))

    reps = commuting_subalgebras(dist, max_dim=2)
    # should have one 2D commuting translation subalgebra representative
    assert any(r.distribution.size == 2 for r in reps)
    # no duplicate signatures
    assert len({r.signature for r in reps}) == len(reps)
