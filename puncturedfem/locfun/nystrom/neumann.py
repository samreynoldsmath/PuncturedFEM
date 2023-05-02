import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from ...mesh.cell import cell
from . import single_layer, double_layer, apply_double_layer

def solve_neumann_zero_average(K: cell, u_wnd):

	# single and double layer operators
	T1 = single_layer.single_layer_mat(K)
	T2 = double_layer.double_layer_mat(K)

	# RHS
	b = T1 @ u_wnd

	#
	T2_sum = np.sum(T2, 1)
	integrator_mat = K.get_integrator()

	# total number of points on the boundary
	N = K.num_pts

	# define linear operator functionality
	def linop4harmconj(u):
		y = apply_double_layer.apply_T2(u, T2, T2_sum, K.closest_vert_idx)
		y += integrator_mat @ u
		return y

	# define linear operator object
	A = LinearOperator(
		dtype = float,
		shape = (N, N),
		matvec = linop4harmconj
	)

	# solve Nystrom system
	u, flag = gmres(A, b, atol=1e-12, tol=1e-12)

	if flag > 0:
		print(f'Something went wrong: GMRES returned flag = {flag}')

	return u