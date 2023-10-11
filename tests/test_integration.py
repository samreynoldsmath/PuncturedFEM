"""
test_integration.py
===================

Test the MeshCell class by computing the length of the boundary of a mesh cell
by integrating 1 over the boundary and comparing to the exact length of the
boundary.

"""

import numpy as np

import puncturedfem as pf

from .build_cell import build_circle, build_square, build_punctured_square
from .build_quad import get_quad_dict

TOL = 1e-10


def compute_boundary_length(
    K: pf.MeshCell, quad_dict: dict[str, pf.Quad]
) -> float:
    """
    Compute the length of the boundary of a mesh cell by integrating 1 over the
    boundary.
    """
    # parameterize edges
    K.parameterize(quad_dict)

    # calculate length of boundary by integrating 1 over boundary
    one = np.ones((K.num_pts,))
    boundary_length = K.integrate_over_boundary(one)

    return boundary_length


def test_integration_circle():
    """
    Test integration over cell boundary of a disk by integrating 1 over boundary
    and comparing to exact length of boundary.
    """

    # set up test parameters
    quad_dict = get_quad_dict()

    # build mesh cell
    K, cell_data = build_circle()

    # compute length of boundary
    boundary_length_computed = compute_boundary_length(K, quad_dict)

    # compute error
    boundary_length_error = np.abs(
        cell_data["boundary_length"] - boundary_length_computed
    )

    # check that computed length is close to exact length
    assert boundary_length_error < TOL


def test_integration_square():
    """
    Test integration over cell boundary of a square by integrating 1 over
    boundary and comparing to exact length of boundary.
    """

    # set up test parameters
    quad_dict = get_quad_dict()

    # build mesh cell
    K, cell_data = build_square()

    # compute length of boundary
    boundary_length_computed = compute_boundary_length(K, quad_dict)

    # compute error
    boundary_length_error = np.abs(
        cell_data["boundary_length"] - boundary_length_computed
    )

    # check that computed length is close to exact length
    assert boundary_length_error < TOL


def test_integration_punctured_square():
    """
    Test integration over cell boundary of a square with a disk removed by
    integrating 1 over boundary and comparing to exact length of boundary.
    """

    # set up test parameters
    quad_dict = get_quad_dict()

    # build mesh cell
    K, cell_data = build_punctured_square()

    # compute length of boundary
    boundary_length_computed = compute_boundary_length(K, quad_dict)

    # compute error
    boundary_length_error = np.abs(
        cell_data["boundary_length"] - boundary_length_computed
    )

    # check that computed length is close to exact length
    assert boundary_length_error < TOL
