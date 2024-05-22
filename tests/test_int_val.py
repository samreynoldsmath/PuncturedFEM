"""
Verify that the interior value computations are accurate.
"""

import numpy as np

import puncturedfem as pf

from .build_cell import build_ghost

TOL = 1e-10


def test_int_val_ghost() -> None:
    """
    Interior value computations for a harmonic function on a ghost cell.

    See also Example 1.3.
    """
    # build mesh cell
    K = build_ghost()

    # set minimum distance to boundary of sampled points
    K.set_interior_point_tolerance(rtol=0.1, atol=0.1)

    # define quadrature schemes
    quad_dict = pf.get_quad_dict(n=64)

    # parameterize edges
    K.parameterize(quad_dict)

    # set up Nyström solver
    nyst = pf.NystromSolver(K, verbose=True)

    # get coordinates of boundary points
    x1, x2 = K.get_boundary_points()

    # trace of v
    v_trace = (
        (x1 - 0.25) / ((x1 - 0.25) ** 2 + (x2 - 0.7) ** 2)
        + (x1**3) * x2
        + x2**2
    )

    # Laplacian of v
    v_laplacian = pf.Polynomial([(6.0, 1, 1), (2.0, 0, 0)])

    # store v as a local function object
    v = pf.LocalPoissonFunction(
        nyst, v_laplacian, v_trace, evaluate_gradient=True
    )

    # computed values
    v_computed = v.int_vals
    v_x1_computed = v.int_grad1
    v_x2_computed = v.int_grad2

    # coordinates of interior points
    y1 = K.int_x1
    y2 = K.int_x2

    # exact values
    v_exact = (
        (y1 - 0.25) / ((y1 - 0.25) ** 2 + (y2 - 0.7) ** 2)
        + (y1**3) * y2
        + y2**2
    )
    v_x1_exact = (
        -((y1 - 0.25) ** 2 - (y2 - 0.7) ** 2)
        / ((y1 - 0.25) ** 2 + (y2 - 0.7) ** 2) ** 2
        + 3 * (y1**2) * y2
    )
    v_x2_exact = (
        -2
        * (y1 - 0.25)
        * (y2 - 0.7)
        / ((y1 - 0.25) ** 2 + (y2 - 0.7) ** 2) ** 2
        + (y1**3)
        + 2 * y2
    )

    # compute errors (log scale)
    v_error = np.log10(np.abs(v_computed - v_exact))
    v_grad_error = (v_x1_computed - v_x1_exact) ** 2 + (
        v_x2_computed - v_x2_exact
    ) ** 2
    v_grad_error = 0.5 * np.log10(v_grad_error)

    # compute maximum pointwise errors
    max_v_error = np.nanmax(v_error, keepdims=False)
    max_v_grad_error = np.nanmax(v_grad_error, keepdims=False)

    assert max_v_error < TOL
    assert max_v_grad_error < TOL
