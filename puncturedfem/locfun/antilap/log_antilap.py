import numpy as np

from ...mesh.cell import cell
from .. import d2n

def get_log_antilap(K: cell):
	"""
	Returns traces of an anti-Laplacian of logarithmic terms on the boundary
		\Lambda(x) = \frac14 |x|^2 (\ln|x|-1)
	"""
	LAM_trace = np.zeros((K.num_pts, K.num_holes))
	for j in range(K.num_holes):

		# xi = K.hole_int_pts[:,j]
		xi = K.components[j + 1].interior_point
		xi = np.array([xi.x, xi.y])

		def LAM(x):
			_, x_xi_norm_sq = d2n.log_terms.shifted_coordinates(x, xi)
			return 0.125 * x_xi_norm_sq * (np.log(x_xi_norm_sq) - 2)

		LAM_trace[:, j] = K.evaluate_function_on_boundary(LAM)

	return LAM_trace

def get_log_antilap_weighted_normal_derivative(K: cell):
	"""
	Returns traces of an anti-Laplacian of logarithmic terms on the boundary:
		\Lambda(x) = \frac14 |x|^2 (\ln|x|-1)
	"""
	dLAM_dn_wgt = np.zeros((K.num_pts, K.num_holes))
	for j in range(K.num_holes):

		# xi = K.hole_int_pts[:,j]
		xi = K.components[j + 1].interior_point
		xi = np.array([xi.x, xi.y])

		def LAM_x1(x):
			x_xi, x_xi_norm_sq = d2n.log_terms.shifted_coordinates(x, xi)
			return 0.25 * (np.log(x_xi_norm_sq) - 1) * x_xi[0]

		def LAM_x2(x):
			x_xi, x_xi_norm_sq = d2n.log_terms.shifted_coordinates(x, xi)
			return 0.25 * (np.log(x_xi_norm_sq) - 1) * x_xi[1]

		LAM_x1_trace = K.evaluate_function_on_boundary(LAM_x1)
		LAM_x2_trace = K.evaluate_function_on_boundary(LAM_x2)

		dLAM_dn = K.dot_with_normal(LAM_x1_trace, LAM_x2_trace)
		dLAM_dn_wgt[:, j] = K.multiply_by_dx_norm(dLAM_dn)

	return dLAM_dn_wgt