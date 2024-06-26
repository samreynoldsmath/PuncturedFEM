"""
Test the LocalPoissonFunction class.

Functions
---------
test_punctured_square
    Sets up the mesh cell K and functions functions v,w as in
    examples/ex1a-square-hole.ipynb
get_l2_and_h1_errors
    Gets the L^2 and H^1 errors for the punctured square
"""

import numpy as np

import puncturedfem as pf

from .build_cell import build_punctured_square


def test_punctured_square() -> None:
    """
    Sets up the mesh cell K and functions functions v,w as in
    examples/ex1a-square-hole.ipynb
    """

    # set up mesh cell
    K, cell_data = build_punctured_square()

    # parameterize edges
    quad_dict = pf.get_quad_dict(n=64)

    l2_error, h1_error = get_l2_and_h1_errors(K, quad_dict, cell_data)

    assert l2_error < 1e-10
    assert h1_error < 1e-10


def get_l2_and_h1_errors(
    K: pf.MeshCell, quad_dict: pf.QuadDict, cell_data: dict
) -> tuple[float, float]:
    """
    Gets the L^2 and H^1 errors for the punctured square
    """
    K.parameterize(quad_dict, compute_interior_points=False)

    # set up Nystrom solver
    nyst = pf.NystromSolver(K)

    # get the coordinates of sampled boundary points
    x1, x2 = K.get_boundary_points()

    # set target value of logarithmic coefficient
    a_exact = 1

    # set point in hole interior
    xi = cell_data["center"]

    # define trace of v
    v_trace = (
        np.exp(x1) * np.cos(x2)
        + 0.5 * a_exact * np.log((x1 - xi[0]) ** 2 + (x2 - xi[1]) ** 2)
        + x1**3 * x2
        + x1 * x2**3
    )

    # create Polynomial object
    v_laplacian = pf.Polynomial([(12.0, 1, 1)])

    # create local function object
    v = pf.LocalPoissonFunction(
        nyst, v_laplacian, v_trace, evaluate_interior=False
    )

    # trace of w
    w_trace = (
        (x1 - 0.5) / ((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2) + x1**3 + x1 * x2**2
    )

    # define a monomial term by specifying its multi-index and coefficient
    w_laplacian = pf.Polynomial([(8.0, 1, 0)])

    # declare w as local function object
    w = pf.LocalPoissonFunction(
        nyst, w_laplacian, w_trace, evaluate_interior=False
    )

    # compute L^2 inner product
    l2_vw_exact = 1.39484950156676
    l2_vw_computed = v.get_l2_inner_prod(w)

    # compute L^2 inner product in a different way
    l2_wv_computed = w.get_l2_inner_prod(v)
    assert abs(l2_vw_computed - l2_wv_computed) < 1e-10

    # compare to exact values
    l2_error = abs(l2_vw_computed - l2_vw_exact)

    # compare to exact values
    h1_vw_exact = 4.46481780319135
    h1_vw_computed = v.get_h1_semi_inner_prod(w)
    h1_error = abs(h1_vw_computed - h1_vw_exact)

    return l2_error, h1_error
