def test_planner_recognizes_advanced_laplace3d_kernel_directly():
    from sympy import DiracDelta, Eq, Function, diff, symbols

    from pdesolve.problem import build_pde_problem

    x, y, z = symbols("x y z", real=True)
    xi, yi, zi = symbols("xi yi zi", real=True)
    u = Function("u")
    eq = Eq(
        diff(u(x, y, z), x, 2) + diff(u(x, y, z), y, 2) + diff(u(x, y, z), z, 2),
        DiracDelta(x - xi) * DiracDelta(y - yi) * DiracDelta(z - zi),
    )
    problem = build_pde_problem(eq, u(x, y, z), (x, y, z))
    kp = problem.details.get("kernel_plan")
    assert kp is not None
    assert kp.operator_family == "laplace_nd"
    assert kp.method == "kernel_green_function" or kp.method == "kernel_fundamental_solution"
