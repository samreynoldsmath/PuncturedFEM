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


def shifted_coordinates(x: np.ndarray, xi: Vert) -> tuple[np.ndarray, float]:
    """
    Returns x - xi and its norm squared
    """
    x_xi = np.array([x[0] - xi.x, x[1] - xi.y])
    x_xi_norm_sq = x_xi[0] ** 2 + x_xi[1] ** 2
    return x_xi, x_xi_norm_sq


def _log_trace(x: np.ndarray, xi: Vert) -> float:
    """
    Returns the trace of the logarithmic term
    """
    _, x_xi_norm_sq = shifted_coordinates(x, xi)
    return 0.5 * np.log(x_xi_norm_sq)


def _log_grad_x1(x: np.ndarray, xi: Vert) -> float:
    """
    Returns the x1 derivative of the logarithmic term
    """
    x_xi, x_xi_norm_sq = shifted_coordinates(x, xi)
    return x_xi[0] / x_xi_norm_sq


def _log_grad_x2(x: np.ndarray, xi: Vert) -> float:
    """
    Returns the x2 derivative of the logarithmic term
    """
    x_xi, x_xi_norm_sq = shifted_coordinates(x, xi)
    return x_xi[1] / x_xi_norm_sq


def get_log_trace(K: MeshCell, interp: int) -> np.ndarray:
    """
    Returns traces of logarithmic terms on the boundary
    """
    lam_trace = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        lam_trace[:, j] = K.evaluate_function_on_boundary(
            fun=partial(_log_trace, xi=K.components[j + 1].interior_point),
            interp=interp,
        )
    return lam_trace


def get_log_grad(K: MeshCell, interp: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns gradients of logarithmic terms on the boundary
    """
    lam_x1_trace = np.zeros((K.num_pts, K.num_holes))
    lam_x2_trace = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        lam_x1_trace[:, j] = K.evaluate_function_on_boundary(
            fun=partial(_log_grad_x1, xi=K.components[j + 1].interior_point),
            interp=interp,
        )
        lam_x2_trace[:, j] = K.evaluate_function_on_boundary(
            fun=partial(_log_grad_x2, xi=K.components[j + 1].interior_point),
            interp=interp,
        )
    return lam_x1_trace, lam_x2_trace


def get_dlam_dt_wgt(
    K: MeshCell, lam_x1_trace: np.ndarray, lam_x2_trace: np.ndarray, interp: int
) -> np.ndarray:
    """
    Returns weighted tangential derivative of logarithmic terms
    """
    dlam_dt_wgt = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        dlam_dt_wgt[:, j] = K.dot_with_tangent(
            lam_x1_trace[:, j], lam_x2_trace[:, j], interp
        )
        dlam_dt_wgt[:, j] = K.multiply_by_dx_norm(dlam_dt_wgt[:, j], interp)
    return dlam_dt_wgt


def get_dlam_dn_wgt(
    K: MeshCell, lam_x1_trace: np.ndarray, lam_x2_trace: np.ndarray, interp: int
) -> np.ndarray:
    """
    Returns weighted normal derivative of logarithmic terms
    """
    dlam_dn_wgt = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        dlam_dn_wgt[:, j] = K.dot_with_normal(
            lam_x1_trace[:, j], lam_x2_trace[:, j], interp
        )
        dlam_dn_wgt[:, j] = K.multiply_by_dx_norm(dlam_dn_wgt[:, j], interp)
    return dlam_dn_wgt
