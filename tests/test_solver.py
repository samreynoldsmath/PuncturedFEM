"""
Test the solver class.

Functions
---------
test_solver_deg1
"""

import numpy as np

import puncturedfem as pf

TOL = 1e-12


def _build_solver(deg: int, n: int) -> pf.Solver:
    # set test parameters
    a = 1.0
    c = 1.0
    f = pf.Polynomial([(1.0, 0, 0)])

    # define bilinear form
    bilinear_form = pf.BilinearForm(
        diffusion_constant=a,
        reaction_constant=c,
        rhs_poly=f,
    )

    # build mesh
    mesh = pf.meshlib.pacman_subdiv(verbose=False)

    # build quadrature dictionary for parameterization
    quad_dict = pf.get_quad_dict(n)

    # build global function space
    glob_fun_sp = pf.GlobalFunctionSpace(mesh, deg, quad_dict, verbose=True)

    # return solver object
    return pf.Solver(
        glob_fun_sp, bilinear_form, verbose=True, compute_interior_values=False
    )


def test_solver_deg_1() -> None:
    """
    Test the Solver class on the Pac-Man-Ghost mesh with degree 1.

    See Also
    --------
    examples/ex2.1-pacman-fem.ipynb
    """
    deg = 1
    n = 32
    filename = f"tests/data/pac_man_ghost_n{n}_deg{deg}.txt"
    x_exact = np.loadtxt(filename)

    solver = _build_solver(deg, n)
    solver.solve()
    x_computed = solver.soln

    x_error = x_exact - x_computed

    norm_error = np.dot(x_error, x_error)
    h1_error = np.dot(solver.stiff_mat @ x_error, x_error)
    l2_error = np.dot(solver.mass_mat @ x_error, x_error)

    assert h1_error > -TOL
    assert l2_error > -TOL
    assert norm_error < TOL
    assert h1_error + l2_error < TOL


def test_solver_deg_2() -> None:
    """
    Test the Solver class on the Pac-Man-Ghost mesh with degree 1.

    See Also
    --------
    examples/ex2.1-pacman-fem.ipynb
    """
    deg = 2
    n = 32
    filename = f"tests/data/pac_man_ghost_n{n}_deg{deg}.txt"
    x_exact = np.loadtxt(filename)

    solver = _build_solver(deg, n)
    solver.solve()
    x_computed = solver.soln

    x_error = x_exact - x_computed

    norm_error = np.dot(x_error, x_error)
    h1_error = np.dot(solver.stiff_mat @ x_error, x_error)
    l2_error = np.dot(solver.mass_mat @ x_error, x_error)

    assert h1_error > -TOL
    assert l2_error > -TOL
    assert norm_error < TOL
    assert h1_error + l2_error < TOL


def test_solver_deg_3() -> None:
    """
    Test the Solver class on the Pac-Man-Ghost mesh with degree 1.

    See Also
    --------
    examples/ex2.1-pacman-fem.ipynb
    """
    deg = 3
    n = 32
    filename = f"tests/data/pac_man_ghost_n{n}_deg{deg}.txt"
    x_exact = np.loadtxt(filename)

    solver = _build_solver(deg, n)
    solver.solve()
    x_computed = solver.soln

    x_error = x_exact - x_computed

    norm_error = np.dot(x_error, x_error)
    h1_error = np.dot(solver.stiff_mat @ x_error, x_error)
    l2_error = np.dot(solver.mass_mat @ x_error, x_error)

    assert h1_error > -TOL
    assert l2_error > -TOL
    assert norm_error < TOL
    assert h1_error + l2_error < TOL
