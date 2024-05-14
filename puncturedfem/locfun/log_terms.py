"""
Logarithmic terms for multiply connected domains.

Let K be a mesh cell with holes. This module provides functions to compute
    lambda(x) = ln|x-xi|
on a multiply connected domain, with xi in the interior of a hole.

Routines in this module
-----------------------
get_log_trace(K)
get_log_grad(K)
get_dlam_dt_wgt(K, lam_x1_trace, lam_x2_trace)
get_dlam_dn_wgt(K, lam_x1_trace, lam_x2_trace)

Notes
-----
- Includes functions to compute traces and derivatives.
- lam denotes lambda, dlam denotes a derivative of lambda.
"""

from functools import partial

import numpy as np

from ..mesh.cell import MeshCell
from ..mesh.vert import Vert


def shifted_coordinates(
    x1: np.ndarray, x2: np.ndarray, xi: Vert
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get x - xi and its norm squared.

    Parameters
    ----------
    x1 : np.ndarray
        x1 coordinate of the points.
    x2 : np.ndarray
        x2 coordinate of the points.
    xi : Vert
        Interior point of the hole.

    Returns
    -------
    x1_xi : np.ndarray
        x_1 - xi_1.
    x2_xi : np.ndarray
        x_2 - xi_2.
    x_xi_norm_sq : np.ndarray
        Norm squared of x - xi.
    """
    x1_xi = x1 - xi.x
    x2_xi = x2 - xi.y
    x_xi_norm_sq = x1_xi**2 + x2_xi**2
    return x1_xi, x2_xi, x_xi_norm_sq


def _log_trace(x1: np.ndarray, x2: np.ndarray, xi: Vert) -> np.ndarray:
    """
    Trace of the logarithmic term.

    Parameters
    ----------
    x1 : np.ndarray
        x1 coordinate of the points.
    x2 : np.ndarray
        x2 coordinate of the points.
    xi : Vert
        Interior point of the hole.

    Returns
    -------
    np.ndarray
        Value of the logarithmic term at the points.
    """
    _, _, x_xi_norm_sq = shifted_coordinates(x1, x2, xi)
    return 0.5 * np.log(x_xi_norm_sq)


def _log_grad_x1(x1: np.ndarray, x2: np.ndarray, xi: Vert) -> np.ndarray:
    """
    Compute derivative of the logarithmic term with respect to x1.

    Parameters
    ----------
    x1 : np.ndarray
        x1 coordinate of the points.
    x2 : np.ndarray
        x2 coordinate of the points.
    xi : Vert
        Interior point of the hole.

    Returns
    -------
    np.ndarray
        Derivative of the logarithmic term with respect to x1.
    """
    x1_xi, _, x_xi_norm_sq = shifted_coordinates(x1, x2, xi)
    return x1_xi / x_xi_norm_sq


def _log_grad_x2(x1: np.ndarray, x2: np.ndarray, xi: Vert) -> np.ndarray:
    """
    Compute derivative of the logarithmic term with respect to x2.

    Parameters
    ----------
    x1 : np.ndarray
        x1 coordinate of the points.
    x2 : np.ndarray
        x2 coordinate of the points.
    xi : Vert
        Interior point of the hole.

    Returns
    -------
    np.ndarray
        Derivative of the logarithmic term with respect to x2.
    """
    _, x2_xi, x_xi_norm_sq = shifted_coordinates(x1, x2, xi)
    return x2_xi / x_xi_norm_sq


def get_log_trace(K: MeshCell) -> np.ndarray:
    """
    Traces of logarithmic terms on the boundary.

    Parameters
    ----------
    K : MeshCell
        Mesh cell with holes.

    Returns
    -------
    np.ndarray
        Traces of the logarithmic terms on the boundary.
    """
    lam_trace = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        lam_trace[:, j] = K.evaluate_function_on_boundary(
            fun=partial(_log_trace, xi=K.components[j + 1].interior_point)
        )
    return lam_trace


def get_log_grad(K: MeshCell) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradients of logarithmic terms on the boundary.

    Parameters
    ----------
    K : MeshCell
        Mesh cell with holes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        First and second components of the gradients of the logarithmic terms.
    """
    lam_x1_trace = np.zeros((K.num_pts, K.num_holes))
    lam_x2_trace = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        lam_x1_trace[:, j] = K.evaluate_function_on_boundary(
            fun=partial(_log_grad_x1, xi=K.components[j + 1].interior_point)
        )
        lam_x2_trace[:, j] = K.evaluate_function_on_boundary(
            fun=partial(_log_grad_x2, xi=K.components[j + 1].interior_point)
        )
    return lam_x1_trace, lam_x2_trace


def get_dlam_dt_wgt(
    K: MeshCell, lam_x1_trace: np.ndarray, lam_x2_trace: np.ndarray
) -> np.ndarray:
    """
    Weighted tangential derivative of logarithmic terms.

    Parameters
    ----------
    K : MeshCell
        Mesh cell with holes.
    lam_x1_trace : np.ndarray
        First component of the gradient of the logarithmic terms.
    lam_x2_trace : np.ndarray
        Second component of the gradient of the logarithmic terms.

    Returns
    -------
    np.ndarray
        Weighted tangential derivative of the logarithmic terms.
    """
    dlam_dt_wgt = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        dlam_dt_wgt[:, j] = K.dot_with_tangent(
            lam_x1_trace[:, j], lam_x2_trace[:, j]
        )
        dlam_dt_wgt[:, j] = K.multiply_by_dx_norm(dlam_dt_wgt[:, j])
    return dlam_dt_wgt


def get_dlam_dn_wgt(
    K: MeshCell, lam_x1_trace: np.ndarray, lam_x2_trace: np.ndarray
) -> np.ndarray:
    """
    Weighted normal derivative of logarithmic terms.

    Parameters
    ----------
    K : MeshCell
        Mesh cell with holes.
    lam_x1_trace : np.ndarray
        First component of the gradient of the logarithmic terms.
    lam_x2_trace : np.ndarray
        Second component of the gradient of the logarithmic terms.

    Returns
    -------
    np.ndarray
        Weighted normal derivative of the logarithmic terms.
    """
    dlam_dn_wgt = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        dlam_dn_wgt[:, j] = K.dot_with_normal(
            lam_x1_trace[:, j], lam_x2_trace[:, j]
        )
        dlam_dn_wgt[:, j] = K.multiply_by_dx_norm(dlam_dn_wgt[:, j])
    return dlam_dn_wgt
