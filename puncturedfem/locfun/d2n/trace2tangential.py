import numpy as np
from ...mesh.cell import cell
from . import fft_deriv

def get_weighted_tangential_derivative_from_trace(
		K: cell, f_vals: np.array) -> np.array:
	"""
	Returns df / ds = \nabla f(x(t)) \cdot x'(t) by computing the derivative
	of f(x(t)) with respect to the scalar parameter t, where x(t)
	is a parameterization of the boundary
	"""

	df_dt_wgt = np.zeros((K.num_pts,))

	for i in range(K.num_holes + 1):

		# get indices of this contour
		j = K.component_start_idx[i]
		jp1 = K.component_start_idx[i + 1]

		# get values on this contour
		f_vals_c = f_vals[j:jp1]

		# compute weighted tangential derivative on this contour
		interval_length = 2 * np.pi * K.components[i].num_edges
		dfc_dt_wgt = fft_deriv.fft_derivative(f_vals_c, interval_length)

		# add to weighted tangential derivative on the whole boundary
		df_dt_wgt[j:jp1] = dfc_dt_wgt

	return df_dt_wgt