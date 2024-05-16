"""
test_solver.py
==============

Test the solver class.
"""

import numpy as np

import puncturedfem as pf

TOL = 1e-12


def _build_solver(deg: int) -> pf.Solver:
    # set test parameters
    n = 16
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
    glob_fun_sp = pf.GlobalFunctionSpace(mesh, deg, quad_dict, verbose=False)

    # return solver object
    return pf.Solver(
        glob_fun_sp, bilinear_form, verbose=False, compute_interior_values=False
    )


def test_solver_deg1() -> None:
    """
    Test the Solver class with degree 1.

    See Also
    --------
    examples/ex2.1-pacman-fem.ipynb
    """
    solver = _build_solver(deg=1)
    solver.solve()

    x_computed = solver.soln
    x_exact = np.loadtxt("tests/data/glob_soln_n16.txt")
    x_error = x_exact - x_computed

    h1_error = np.dot(solver.stiff_mat @ x_error, x_error)
    l2_error = np.dot(solver.mass_mat @ x_error, x_error)

    assert h1_error + l2_error < TOL
