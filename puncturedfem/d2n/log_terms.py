import numpy as np

from ..mesh import *

def shifted_coordinates(x, xi):
	x_xi = np.array([x[0] - xi[0], x[1] - xi[1]])
	x_xi_norm_sq = x_xi[0]**2 + x_xi[1]**2
	return x_xi, x_xi_norm_sq

def get_log_trace(K: cell):
	"""
	Returns traces of logarithmic terms on the boundary
	"""
	lam_trace = np.zeros((K.num_pts, K.num_holes))
	for j in range(K.num_holes):

		xi = K.hole_int_pts[:,j]

		def lam(x):
			x_xi, x_xi_norm_sq = shifted_coordinates(x, xi)
			return 0.5 * np.log(x_xi_norm_sq)

		lam_trace[:, j] = K.evaluate_function_on_boundary(lam)

	return lam_trace

def get_log_grad(K: cell):
	"""
	Returns gradients of logarithmic terms on the boundary
	"""
	lam_x1_trace = np.zeros((K.num_pts, K.num_holes))
	lam_x2_trace = np.zeros((K.num_pts, K.num_holes))

	for j in range(K.num_holes):

		xi = K.hole_int_pts[:,j]

		def lam_x1(x):
			x_xi, x_xi_norm_sq = shifted_coordinates(x, xi)
			return x_xi[0] / x_xi_norm_sq

		def lam_x2(x):
			x_xi, x_xi_norm_sq = shifted_coordinates(x, xi)
			return x_xi[1] / x_xi_norm_sq

		lam_x1_trace[:, j] = K.evaluate_function_on_boundary(lam_x1)
		lam_x2_trace[:, j] = K.evaluate_function_on_boundary(lam_x2)

	return lam_x1_trace, lam_x2_trace

def get_dlam_dt_wgt(K, lam_x1_trace, lam_x2_trace):
	"""
	Returns weighted tangential derivative of logarthmic terms
	"""
	dlam_dt_wgt = np.zeros((K.num_pts, K.num_holes))
	for j in range(K.num_holes):
		dlam_dt_wgt[:, j] \
			= K.dot_with_tangent(lam_x1_trace[:, j], lam_x2_trace[:, j])
		dlam_dt_wgt[:, j] = K.multiply_by_dx_norm(dlam_dt_wgt[:, j])
	return dlam_dt_wgt

def get_dlam_dn_wgt(K, lam_x1_trace, lam_x2_trace):
	"""
	Returns weighted normal derivative of logarthmic terms
	"""
	dlam_dn_wgt = np.zeros((K.num_pts, K.num_holes))
	for j in range(K.num_holes):
		dlam_dn_wgt[:, j] \
			= K.dot_with_normal(lam_x1_trace[:, j], lam_x2_trace[:, j])
		dlam_dn_wgt[:, j] = K.multiply_by_dx_norm(dlam_dn_wgt[:, j])
	return dlam_dn_wgt