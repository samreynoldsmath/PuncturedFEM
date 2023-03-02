import numpy as np
from ..mesh.cell import cell
from ..d2n.fft_deriv import fft_antiderivative
from ..d2n.log_terms import get_log_grad
from . import log_antilap
from .. import nystrom

def get_anti_laplacian_harmonic(K: cell, psi, psi_hat, a):
	"""
	Returns the trace and weighted normal derivative of an anti-Laplacian of a
	harmonic function
		phi = psi + sum_{j=1}^m a_j ln |x - xi_j|
	given the trace of psi, the trace of its harmonic conjugate psi_hat,
	and the logarithmic coefficients a_1,...,a_m
	(When K is simply connected, phi = psi and a is an empty list)
	"""

	if K.num_holes == 0:
		PHI, PHI_wnd = _antilap_simply_connected(K, psi, psi_hat)
	else:
		PHI, PHI_wnd = _antilap_multiply_connected(K, psi, psi_hat, a)

	return PHI, PHI_wnd

def _antilap_simply_connected(K: cell, phi, phi_hat):

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
	PHI_x1 = 0.25 * (rho 	 + x1 * phi + x2 * phi_hat)
	PHI_x2 = 0.25 * (rho_hat + x2 * phi - x1 * phi_hat)

	# weigthed normal derivative of anti-Laplacian
	PHI_nd = K.dot_with_normal(PHI_x1, PHI_x2)
	PHI_wnd = K.multiply_by_dx_norm(PHI_nd)

	return PHI, PHI_wnd

def _antilap_multiply_connected(K: cell, psi, psi_hat, a):

	# compute F * t and \hat F * t
	F_t = K.dot_with_tangent(psi, -psi_hat)
	F_hat_t = K.dot_with_tangent(psi_hat, psi)

	# compute b_j and c_j
	b = np.zeros((K.num_holes,))
	c = np.zeros((K.num_holes,))
	for j in range(K.num_holes):
		b[j] = K.integrate_over_specific_contour(F_hat_t, j + 1) / (-2 * np.pi)
		c[j] = K.integrate_over_specific_contour(F_t, j + 1) / (2 * np.pi)

	# compute \mu_j and \hat\mu_j
	mu, mu_hat = get_log_grad(K)
	mu_hat *= -1

	# compute \psi_0 and \hat\psi_0
	psi_0 = psi - (mu @ b - mu_hat @ c)
	psi_hat_0 = psi_hat - (mu @ c + mu_hat @ b)

	# compute weighted normal derivatives of rho and rho_hat
	rho_nd_0 = K.dot_with_normal(psi_0, -psi_hat_0)
	rho_wnd_0 = K.multiply_by_dx_norm(rho_nd_0)
	rho_hat_nd_0 = K.dot_with_normal(psi_hat_0, psi_0)
	rho_hat_wnd_0 = K.multiply_by_dx_norm(rho_hat_nd_0)

	# solve for rho_0 and rho_hat_0
	rho_0 = nystrom.neumann.solve_neumann_zero_average(K, rho_wnd_0)
	rho_hat_0 = nystrom.neumann.solve_neumann_zero_average(K, rho_hat_wnd_0)

	# compute anti-Laplacian of psi_0
	x1, x2 = K.get_boundary_points()
	PHI = 0.25 * (x1 * rho_0 + x2 * rho_hat_0)
	PHI_x1 = 0.25 * (rho_0     + x1 * psi_0 + x2 * psi_hat_0)
	PHI_x2 = 0.25 * (rho_hat_0 + x2 * psi_0 - x1 * psi_hat_0)
	PHI_nd = K.dot_with_normal(PHI_x1, PHI_x2)
	PHI_wnd = K.multiply_by_dx_norm(PHI_nd)

	# compute M = sum_j M_j
	for j in range(K.num_holes):
		xi = K.hole_int_pts[:, j]
		x_xi_1 = x1 - xi[0]
		x_xi_2 = x2 - xi[1]
		x_xi_norm_sq = x_xi_1 ** 2 + x_xi_2 ** 2
		log_x_xi_norm = 0.5 * np.log(x_xi_norm_sq)
		PHI += 0.5 * (b[j] * x_xi_1 + c[j] * x_xi_2) * log_x_xi_norm
		M_x1 = 0.5 * (b[j] * mu[:, j] - c[j] * mu_hat[:, j]) *x_xi_1 \
			+ 0.5 * b[j] * log_x_xi_norm
		M_x2 = 0.5 * (b[j] * mu[:, j] - c[j] * mu_hat[:, j]) *x_xi_2 \
			+ 0.5 * c[j] * log_x_xi_norm
		M_nd = K.dot_with_normal(M_x1, M_x2)
		PHI_wnd += K.multiply_by_dx_norm(M_nd)

	# compute \Lambda_j
	PHI += log_antilap.get_log_antilap(K) @ a
	PHI_wnd += log_antilap.get_log_antilap_weighted_normal_derivative(K) @ a

	return PHI, PHI_wnd

def _rational_function_coefficients(K: cell, F_t, F_hat_t):
	# TODO
	pass

def _antilap_rational_terms(K:cell, b, c):
	# TODO
	pass

def _antilap_log_terms(K: cell):
	# TODO
	pass