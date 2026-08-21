import sympy as sp

from pdesolve.classical import (
    classify_linear_second_order_pde,
    pdesolve,
)


def test_classify_linear_second_order_pde_2vars_wave_heat_laplace():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")

    wave = classify_linear_second_order_pde(
        sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)), u(x, t)
    )
    heat = classify_linear_second_order_pde(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)), u(x, t)
    )
    laplace = classify_linear_second_order_pde(
        sp.Eq(sp.diff(u(x, t), x, 2) + sp.diff(u(x, t), t, 2), 0), u(x, t)
    )

    assert wave.classification == "hyperbolic"
    assert heat.classification == "parabolic"
    assert laplace.classification == "elliptic"


def test_classify_linear_second_order_pde_3vars_wave_and_ultrahyperbolic():
    x, y, z = sp.symbols("x y z", real=True)
    u = sp.Function("u")

    wave3 = classify_linear_second_order_pde(
        sp.Eq(
            sp.diff(u(x, y, z), z, 2),
            sp.diff(u(x, y, z), x, 2) + sp.diff(u(x, y, z), y, 2),
        ),
        u(x, y, z),
    )
    ultra = classify_linear_second_order_pde(
        sp.Eq(
            sp.diff(u(x, y, z), x, 2)
            + sp.diff(u(x, y, z), y, 2)
            - sp.diff(u(x, y, z), z, 2),
            0,
        ),
        u(x, y, z),
    )

    assert wave3.classification == "hyperbolic"
    assert ultra.classification == "hyperbolic"


def test_classify_linear_second_order_pde_4vars_ultrahyperbolic():
    w, x, y, z = sp.symbols("w x y z", real=True)
    u = sp.Function("u")
    eq = sp.Eq(
        sp.diff(u(w, x, y, z), w, 2)
        + sp.diff(u(w, x, y, z), x, 2)
        - sp.diff(u(w, x, y, z), y, 2)
        - sp.diff(u(w, x, y, z), z, 2),
        0,
    )
    cls = classify_linear_second_order_pde(eq, u(w, x, y, z))
    assert cls.classification == "ultrahyperbolic"


def test_pdesolve_auto_uses_general_classifier_for_wave_and_heat():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")

    wave_res = pdesolve(
        sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
        u(x, t),
        ics={"initial_displacement": lambda z: z, "initial_velocity": lambda z: 1},
        method="auto",
    )
    heat_res = pdesolve(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        ics={"initial_profile": lambda z: z * (sp.pi - z)},
        bcs={"type": "dirichlet_homogeneous_interval", "length": sp.pi, "terms": 3},
        method="auto",
    )

    assert wave_res.method == "dAlembert_wave_ivp"
    assert heat_res.method == "heat_dirichlet_sine_series"
