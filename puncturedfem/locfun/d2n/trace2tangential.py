import numpy as np

from ...mesh.cell import cell
from ...mesh.contour import contour
from . import fft_deriv

def get_weighted_tangential_derivative_from_trace(K: cell, f):
	"""
	Returns df / ds = \nabla f(x(t)) \cdot x'(t) by computing the derivative
	of f(x(t)) with respect to the scalar parameter t, where x(t)
	is a parameterization of the boundary
	"""

	N = K.num_pts
	df_dt_wgt = np.zeros((N,))

	for c_idx in K.contour_idx:

		# create contour object
		c = contour([K.edge_list[i] for i in c_idx])

		# get values of f on c
		fc = np.zeros((c.num_pts,))
		for i in range(c.num_edges):
			fc[c.vert_idx[i]:c.vert_idx[i + 1]] = \
				f[K.vert_idx[c_idx[i]]:K.vert_idx[c_idx[i] + 1]]

		# compute weighted tangential derivative on this contour
		interval_length = 2 * np.pi * c.num_edges
		dfc_dt_wgt = fft_deriv.fft_derivative(fc, interval_length)

		# assign weighted tangential derivative to position in K
		fc = np.zeros((c.num_pts,))
		for i in range(c.num_edges):
			df_dt_wgt[K.vert_idx[c_idx[i]]:K.vert_idx[c_idx[i] + 1]] = \
				dfc_dt_wgt[c.vert_idx[i]:c.vert_idx[i + 1]]

	return df_dt_wgt