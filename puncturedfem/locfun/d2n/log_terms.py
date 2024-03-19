"""
log_terms.py
============

Logarithmic terms of the form
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

from ...mesh.cell import MeshCell
from ...mesh.vert import Vert


def shifted_coordinates(
    x1: np.ndarray, x2: np.ndarray, xi: Vert
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns x - xi and its norm squared
    """
    x1_xi = x1 - xi.x
    x2_xi = x2 - xi.y
    x_xi_norm_sq = x1_xi**2 + x2_xi**2
    return x1_xi, x2_xi, x_xi_norm_sq


def _log_trace(x1: np.ndarray, x2: np.ndarray, xi: Vert) -> np.ndarray:
    """
    Returns the trace of the logarithmic term
    """
    _, _, x_xi_norm_sq = shifted_coordinates(x1, x2, xi)
    return 0.5 * np.log(x_xi_norm_sq)


def _log_grad_x1(x1: np.ndarray, x2: np.ndarray, xi: Vert) -> np.ndarray:
    """
    Returns the x1 derivative of the logarithmic term
    """
    x1_xi, _, x_xi_norm_sq = shifted_coordinates(x1, x2, xi)
    return x1_xi / x_xi_norm_sq


def _log_grad_x2(x1: np.ndarray, x2: np.ndarray, xi: Vert) -> np.ndarray:
    """
    Returns the x2 derivative of the logarithmic term
    """
    _, x2_xi, x_xi_norm_sq = shifted_coordinates(x1, x2, xi)
    return x2_xi / x_xi_norm_sq


def get_log_trace(K: MeshCell) -> np.ndarray:
    """
    Returns traces of logarithmic terms on the boundary
    """
    lam_trace = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        lam_trace[:, j] = K.evaluate_function_on_boundary(
            fun=partial(_log_trace, xi=K.components[j + 1].interior_point)
        )
    return lam_trace


def get_log_grad(K: MeshCell) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns gradients of logarithmic terms on the boundary
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
    Returns weighted tangential derivative of logarithmic terms
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
    Returns weighted normal derivative of logarithmic terms
    """
    dlam_dn_wgt = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        dlam_dn_wgt[:, j] = K.dot_with_normal(
            lam_x1_trace[:, j], lam_x2_trace[:, j]
        )
        dlam_dn_wgt[:, j] = K.multiply_by_dx_norm(dlam_dn_wgt[:, j])
    return dlam_dn_wgt
