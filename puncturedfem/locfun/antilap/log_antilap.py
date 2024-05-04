"""
Anti-Laplacian of a logarithmic term on am multiply connected domain.

Routines in this module
-----------------------
get_log_antilap(K)
get_log_antilap_weighted_normal_derivative(K)

Notes
-----
Assumes that the interior point of each hole has been set to K.hole_int_pts.
"""

from functools import partial

import numpy as np

from ...mesh.cell import MeshCell
from ...mesh.vert import Vert
from .. import d2n


def get_log_antilap(K: MeshCell) -> np.ndarray:
    """
    Get an anti-Laplacian of logarithmic terms on the boundary.

    Parameters
    ----------
    K : MeshCell
        Mesh cell object.

    Returns
    -------
    np.ndarray
        Traces of an anti-Laplacian of logarithmic terms on the boundary:
        Lambda(x) = 1/4 |x|^2 (ln|x|-1)
    """
    LAM_trace = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        LAM_trace[:, j] = K.evaluate_function_on_boundary(
            fun=partial(_log_antilap, xi=K.components[j + 1].interior_point)
        )
    return LAM_trace


def get_log_antilap_weighted_normal_derivative(K: MeshCell) -> np.ndarray:
    """
    Get the weighted normal derivative of an anti-Laplacian.

    Parameters
    ----------
    K : MeshCell
        Mesh cell object.

    Returns
    -------
    np.ndarray
        Weighted normal derivative of an anti-Laplacian of logarithmic terms on the boundary
    """
    dLAM_dn_wgt = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        LAM_x1_trace = K.evaluate_function_on_boundary(
            fun=partial(_log_antilap_x1, xi=K.components[j + 1].interior_point)
        )
        LAM_x2_trace = K.evaluate_function_on_boundary(
            fun=partial(_log_antilap_x2, xi=K.components[j + 1].interior_point)
        )

        dLAM_dn = K.dot_with_normal(LAM_x1_trace, LAM_x2_trace)
        dLAM_dn_wgt[:, j] = K.multiply_by_dx_norm(dLAM_dn)

    return dLAM_dn_wgt


def _log_antilap(x1: np.ndarray, x2: np.ndarray, xi: Vert) -> np.ndarray:
    """
    Get an anti-Laplacian of a logarithmic term.

    Parameters
    ----------
    x1 : np.ndarray
        x1 coordinates.
    x2 : np.ndarray
        x2 coordinates.
    xi : Vert
        Interior point of the hole.

    Returns
    -------
    np.ndarray
        Values of an anti-Laplacian of the logarithmic term.
    """
    _, _, x_xi_norm_sq = d2n.log_terms.shifted_coordinates(x1, x2, xi)
    return 0.125 * x_xi_norm_sq * (np.log(x_xi_norm_sq) - 2)


def _log_antilap_x1(x1: np.ndarray, x2: np.ndarray, xi: Vert) -> np.ndarray:
    """
    Get the x1 derivative of an anti-Laplacian of the logarithmic term.

    Parameters
    ----------
    x1 : np.ndarray
        x1 coordinates.
    x2 : np.ndarray
        x2 coordinates.
    xi : Vert
        Interior point of the hole.

    Returns
    -------
    np.ndarray
        x1 derivative of an anti-Laplacian of the logarithmic term.
    """
    x1_xi, _, x_xi_norm_sq = d2n.log_terms.shifted_coordinates(x1, x2, xi)
    return 0.25 * (np.log(x_xi_norm_sq) - 1) * x1_xi


def _log_antilap_x2(x1: np.ndarray, x2: np.ndarray, xi: Vert) -> np.ndarray:
    """
    Get the x2 derivative of an anti-Laplacian of the logarithmic term.

    Parameters
    ----------
    x1 : np.ndarray
        x1 coordinates.
    x2 : np.ndarray
        x2 coordinates.
    xi : Vert
        Interior point of the hole.

    Returns
    -------
    np.ndarray
        x2 derivative of an anti-Laplacian of the logarithmic term.
    """
    _, x2_xi, x_xi_norm_sq = d2n.log_terms.shifted_coordinates(x1, x2, xi)
    return 0.25 * (np.log(x_xi_norm_sq) - 1) * x2_xi
