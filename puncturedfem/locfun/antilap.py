"""
Anti-Laplacian of a harmonic function.

Routines in this module
-----------------------
get_anti_laplacian_harmonic(K, psi, psi_hat, log_coef)
_antilap_simply_connected(K, phi, phi_hat)
_antilap_multiply_connected(K, psi, psi_hat, log_coef)
"""

import numpy as np

from .fft_deriv import fft_antiderivative, fft_antiderivative_on_each_component
from .nystrom import NystromSolver
from ..mesh.cell import MeshCell
from .trace import DirichletTrace


def get_anti_laplacian_harmonic(
    nyst: NystromSolver,
    psi: np.ndarray,
    psi_hat: np.ndarray,
    log_coef: np.ndarray,
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
    log_coef : np.ndarray
        Logarithmic weights.

    Returns
    -------
    PHI : np.ndarray
        Trace of the anti-Laplacian.
    PHI_wnd : np.ndarray
        Weighted normal derivative of the anti-Laplacian.
    """
    if nyst.K.num_holes == 0:
        PHI, PHI_wnd = _antilap_simply_connected(nyst.K, psi, psi_hat)
    else:
        PHI, PHI_wnd = _antilap_multiply_connected(nyst, psi, psi_hat, log_coef)

    return PHI, PHI_wnd


def _antilap_simply_connected(
    K: MeshCell, phi: np.ndarray, phi_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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

    return _antilap_from_components(K, rho, rho_hat, phi, phi_hat)


def _antilap_from_components(
    K: MeshCell,
    rho: np.ndarray,
    rho_hat: np.ndarray,
    phi: np.ndarray,
    phi_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    nyst: NystromSolver,
    psi: np.ndarray,
    psi_hat: np.ndarray,
    log_coef: np.ndarray,
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

    # boundary points
    x1, x2 = K.get_boundary_points()

    # compute mu_j and hat_mu_j
    mu = np.zeros((K.num_pts, K.num_holes))
    mu_hat = np.zeros((K.num_pts, K.num_holes))
    for j in range(K.num_holes):
        xi = K.components[j + 1].interior_point
        y1 = x1 - xi.x
        y2 = x2 - xi.y
        y_norm_sq = y1**2 + y2**2
        mu[:, j] = y1 / y_norm_sq
        mu_hat[:, j] = -y2 / y_norm_sq

    # compute psi_0 and hat_psi_0
    psi_0 = psi - (mu @ b - mu_hat @ c)
    psi_hat_0 = psi_hat - (mu @ c + mu_hat @ b)

    # initialize traces of rho and rho_hat
    rho_0 = DirichletTrace(edges=K.get_edges(), values=0)
    rho_hat_0 = DirichletTrace(edges=K.get_edges(), values=0)

    # compute weighted normal derivatives of rho_0 and rho_hat_0
    rho_0 = _compute_wnd_from_gradient(rho_0, psi_0, -psi_hat_0, K)
    rho_hat_0 = _compute_wnd_from_gradient(rho_hat_0, psi_hat_0, psi_0, K)

    # compute rho_0 and rho_hat_0
    if nyst.antilap_strategy == "direct":
        rho_0, rho_hat_0 = _get_rho_and_rho_hat_with_direct_solve(
            nyst, rho_0, rho_hat_0
        )
    elif nyst.antilap_strategy == "fft":
        # comptue weighted tangential derivatives of rho_0 and rho_hat_0
        rho_0 = _compute_wtd_from_gradient(rho_0, psi_0, -psi_hat_0, K)
        rho_hat_0 = _compute_wtd_from_gradient(rho_hat_0, psi_hat_0, psi_0, K)
        # compute rho_0 and rho_hat_0
        rho_0, rho_hat_0 = _get_rho_and_rho_hat_with_fft(nyst, rho_0, rho_hat_0)

    # compute anti-Laplacian of psi_0
    PHI, PHI_wnd = _antilap_from_components(
        K, rho_0.values, rho_hat_0.values, psi_0, psi_hat_0
    )

    # anti-Laplacians of rational and logarithmic terms
    for j in range(K.num_holes):
        # shift coordinates
        xi = K.components[j + 1].interior_point
        y1 = x1 - xi.x
        y2 = x2 - xi.y
        y_norm_sq = y1**2 + y2**2
        log_y_norm_sq = 0.5 * np.log(y_norm_sq)

        # anti-Laplacian of rational terms
        PHI += 0.5 * (b[j] * y1 + c[j] * y2) * log_y_norm_sq
        M_x1 = (
            0.5 * (b[j] * mu[:, j] - c[j] * mu_hat[:, j]) * y1
            + 0.5 * b[j] * log_y_norm_sq
        )
        M_x2 = (
            0.5 * (b[j] * mu[:, j] - c[j] * mu_hat[:, j]) * y2
            + 0.5 * c[j] * log_y_norm_sq
        )
        M_nd = K.dot_with_normal(M_x1, M_x2)
        PHI_wnd += K.multiply_by_dx_norm(M_nd)

        # anti-Laplacian of log terms
        LAM = y_norm_sq * (log_y_norm_sq - 1) / 4
        LAM_x1 = y1 * (2 * log_y_norm_sq - 1) / 4
        LAM_x2 = y2 * (2 * log_y_norm_sq - 1) / 4
        LAM_nd = K.dot_with_normal(LAM_x1, LAM_x2)

        # add contribution to anti-Laplacian
        PHI += log_coef[j] * LAM
        PHI_wnd += log_coef[j] * K.multiply_by_dx_norm(LAM_nd)

    return PHI, PHI_wnd


def _compute_wnd_from_gradient(
    trace: DirichletTrace, grad1: np.ndarray, grad2: np.ndarray, K: MeshCell
) -> DirichletTrace:
    nd = K.dot_with_normal(grad1, grad2)
    trace.set_weighted_normal_derivative(K.multiply_by_dx_norm(nd))
    return trace


def _compute_wtd_from_gradient(
    trace: DirichletTrace, grad1: np.ndarray, grad2: np.ndarray, K: MeshCell
) -> DirichletTrace:
    td = K.dot_with_tangent(grad1, grad2)
    trace.set_weighted_tangential_derivative(K.multiply_by_dx_norm(td))
    return trace


def _get_rho_and_rho_hat_with_direct_solve(
    nyst: NystromSolver, rho: DirichletTrace, rho_hat: DirichletTrace
) -> tuple[DirichletTrace, DirichletTrace]:
    # 1. classic strategy
    #   a. compute weighted normal derivatives of rho and rho_hat
    #   b. solve for rho_0 and rho_hat_0 using nyst.solve_neumann_zero_average
    if rho.w_norm_deriv is None or rho_hat.w_norm_deriv is None:
        raise ValueError("Weighted normal derivatives are not set.")
    rho.set_trace_values(
        values=nyst.solve_neumann_zero_average(rho.w_norm_deriv)
    )
    rho_hat.set_trace_values(
        values=nyst.solve_neumann_zero_average(rho_hat.w_norm_deriv)
    )
    return rho, rho_hat


def _get_rho_and_rho_hat_with_fft(
    nyst: NystromSolver, rho: DirichletTrace, rho_hat: DirichletTrace
) -> tuple[DirichletTrace, DirichletTrace]:
    # 2. new strategy
    #   a. compute weighted normal and tangential derivatives of rho
    #   b. assume rho has an average value of zero on the outer boundary
    #   c. compute trace of omega, a harmonic function whose tangential
    #      derivative is that of rho and whose average on each component of the
    #      boundary is zero, using FFT
    #   d. on the jth component of the boundary (j > 0) let eta_j be a harmonic
    #      function having a normal derivative equal to one on the jth
    #      component, - |partial K_j| / |partial K_)| on the outer boundary, and
    #      zero everywhere else
    #   e. on the jth component of the boundary (j > 0) compute the constants
    #      of integration for the harmonic function omega using
    #      d_j = (1 / |partial K_j|) * int_{partial K} eta_j * (d rho / dn) ds
    #   f. repeat steps a-e for rho_hat
    msg = "Weighted normal and tangential derivatives are not set."
    if rho.w_norm_deriv is None or rho.w_tang_deriv is None:
        raise ValueError(msg)
    if rho_hat.w_norm_deriv is None or rho_hat.w_tang_deriv is None:
        raise ValueError(msg)

    # recover trace of rho from tangential and normal derivatives
    rho = _recover_trace_from_tangential_and_normal_derivatives(nyst, rho)

    # recover trace of rho_hat from tangential and normal derivatives
    rho_hat = _recover_trace_from_tangential_and_normal_derivatives(
        nyst, rho_hat
    )

    return rho, rho_hat


def _recover_trace_from_tangential_and_normal_derivatives(
    nyst: NystromSolver, rho: DirichletTrace
) -> DirichletTrace:
    if nyst.eta is None:
        raise ValueError("Nystrom solver object has no eta functions.")
    if rho.w_norm_deriv is None:
        raise ValueError("Weighted normal derivative is not set.")
    if rho.w_tang_deriv is None:
        raise ValueError("Weighted tangential derivative is not set.")

    rho_vals = fft_antiderivative_on_each_component(nyst.K, rho.w_tang_deriv)
    rho.set_trace_values(values=rho_vals)

    # set integration constants on each component of the boundary
    for j, c in enumerate(nyst.K.components):
        # compute the arc length of the jth component
        arc_length = c.integrate_over_closed_contour(np.ones((c.num_pts,)))

        # determine the average value of rho on the jth component
        start = nyst.K.component_start_idx[j]
        pt_idx = slice(start, start + c.num_pts)
        int_c_rho = c.integrate_over_closed_contour(rho.values[pt_idx])

        # set average value of rho on this component to zero
        rho.values[pt_idx] -= int_c_rho / arc_length

        if j == 0:
            continue

        # compute H^1 semi-inner product of rho with eta_j
        integrand = nyst.eta[j - 1].values * rho.w_norm_deriv
        d_j = nyst.K.integrate_over_boundary_preweighted(integrand)

        # normalize the constant of integration
        d_j /= arc_length

        # set integration constant
        rho.values[pt_idx] += d_j

    return rho
