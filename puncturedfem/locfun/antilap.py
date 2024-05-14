"""
Anti-Laplacian of a harmonic function.

Routines in this module
-----------------------
get_anti_laplacian_harmonic(K, psi, psi_hat, a)
_antilap_simply_connected(K, phi, phi_hat)
_antilap_multiply_connected(K, psi, psi_hat, a)
"""

import numpy as np

from . import log_antilap
from .fft_deriv import fft_antiderivative
from .log_terms import get_log_grad
from .nystrom import NystromSolver


def get_anti_laplacian_harmonic(
    nyst: NystromSolver, psi: np.ndarray, psi_hat: np.ndarray, a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the trace and weighted normal derivative of an anti-Laplacian.

    Parameters
    ----------
    nyst : NystromSolver
        Nystrom solver object.
    psi : np.ndarray
        Trace of a harmonic function.
    psi_hat : np.ndarray
        Trace of its harmonic conjugate.
    a : np.ndarray
        Logarithmic weights.

    Returns
    -------
    PHI : np.ndarray
        Trace of the anti-Laplacian.
    PHI_wnd : np.ndarray
        Weighted normal derivative of the anti-Laplacian.
    """
    if nyst.K.num_holes == 0:
        PHI, PHI_wnd = _antilap_simply_connected(nyst, psi, psi_hat)
    else:
        PHI, PHI_wnd = _antilap_multiply_connected(nyst, psi, psi_hat, a)

    return PHI, PHI_wnd


def _antilap_simply_connected(
    nyst: NystromSolver, phi: np.ndarray, phi_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    K = nyst.K

    # length of interval of integration in parameter space
    interval_length = 2 * np.pi * K.num_edges

    # integrate tangential derivative of rho
    rho_td = K.dot_with_tangent(phi, -phi_hat)
    rho_wtd = K.multiply_by_dx_norm(rho_td)
    rho = fft_antiderivative(rho_wtd, interval_length)

    # integrate tangential derivative of rho_hat
    rho_hat_td = K.dot_with_tangent(phi_hat, phi)
    rho_hat_wtd = K.multiply_by_dx_norm(rho_hat_td)
    rho_hat = fft_antiderivative(rho_hat_wtd, interval_length)

    # coordinates of boundary points
    x1, x2 = K.get_boundary_points()

    # construct anti-Laplacian
    PHI = 0.25 * (x1 * rho + x2 * rho_hat)

    # gradient of anti-Laplacian
    PHI_x1 = 0.25 * (rho + x1 * phi + x2 * phi_hat)
    PHI_x2 = 0.25 * (rho_hat + x2 * phi - x1 * phi_hat)

    # weighted normal derivative of anti-Laplacian
    PHI_nd = K.dot_with_normal(PHI_x1, PHI_x2)
    PHI_wnd = K.multiply_by_dx_norm(PHI_nd)

    return PHI, PHI_wnd


def _antilap_multiply_connected(
    nyst: NystromSolver, psi: np.ndarray, psi_hat: np.ndarray, a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    K = nyst.K

    # compute F * t and \hat F * t
    F_t = K.dot_with_tangent(psi, -psi_hat)
    F_hat_t = K.dot_with_tangent(psi_hat, psi)

    # compute b_j and c_j
    b = np.zeros((K.num_holes,))
    c = np.zeros((K.num_holes,))
    for j in range(K.num_holes):
        k = K.component_start_idx[j + 1]
        kp1 = K.component_start_idx[j + 2]
        F_hat_t_j = F_hat_t[k:kp1]
        F_t_j = F_t[k:kp1]
        b[j] = K.components[j + 1].integrate_over_closed_contour(F_hat_t_j)
        c[j] = K.components[j + 1].integrate_over_closed_contour(F_t_j)
    b /= -2 * np.pi
    c /= 2 * np.pi

    # compute mu_j and hat_mu_j
    mu, mu_hat = get_log_grad(K)
    mu_hat *= -1

    # compute psi_0 and hat_psi_0
    psi_0 = psi - (mu @ b - mu_hat @ c)
    psi_hat_0 = psi_hat - (mu @ c + mu_hat @ b)

    # compute weighted normal derivatives of rho and rho_hat
    rho_nd_0 = K.dot_with_normal(psi_0, -psi_hat_0)
    rho_wnd_0 = K.multiply_by_dx_norm(rho_nd_0)
    rho_hat_nd_0 = K.dot_with_normal(psi_hat_0, psi_0)
    rho_hat_wnd_0 = K.multiply_by_dx_norm(rho_hat_nd_0)

    # solve for rho_0 and rho_hat_0
    rho_0 = nyst.solve_neumann_zero_average(rho_wnd_0)
    rho_hat_0 = nyst.solve_neumann_zero_average(rho_hat_wnd_0)

    # compute anti-Laplacian of psi_0
    x1, x2 = K.get_boundary_points()

    PHI = 0.25 * (x1 * rho_0 + x2 * rho_hat_0)
    PHI_x1 = 0.25 * (rho_0 + x1 * psi_0 + x2 * psi_hat_0)
    PHI_x2 = 0.25 * (rho_hat_0 + x2 * psi_0 - x1 * psi_hat_0)
    PHI_nd = K.dot_with_normal(PHI_x1, PHI_x2)
    PHI_wnd = K.multiply_by_dx_norm(PHI_nd)

    # compute M = sum_j M_j
    for j in range(K.num_holes):
        xi = K.components[j + 1].interior_point
        x_xi_1 = np.array(x1 - xi.x)
        x_xi_2 = np.array(x2 - xi.y)
        x_xi_norm_sq = x_xi_1**2 + x_xi_2**2
        log_x_xi_norm = 0.5 * np.log(x_xi_norm_sq)
        PHI += 0.5 * (b[j] * x_xi_1 + c[j] * x_xi_2) * log_x_xi_norm
        M_x1 = (
            0.5 * (b[j] * mu[:, j] - c[j] * mu_hat[:, j]) * x_xi_1
            + 0.5 * b[j] * log_x_xi_norm
        )
        M_x2 = (
            0.5 * (b[j] * mu[:, j] - c[j] * mu_hat[:, j]) * x_xi_2
            + 0.5 * c[j] * log_x_xi_norm
        )
        M_nd = K.dot_with_normal(M_x1, M_x2)
        PHI_wnd += K.multiply_by_dx_norm(M_nd)

    # compute Lambda_j
    PHI += log_antilap.get_log_antilap(K) @ a
    PHI_wnd += log_antilap.get_log_antilap_weighted_normal_derivative(K) @ a

    return PHI, PHI_wnd
