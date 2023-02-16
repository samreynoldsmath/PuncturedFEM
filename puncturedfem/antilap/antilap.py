import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from .. import mesh
from .. import d2n
from .. import nystrom

def get_anti_laplacian_harmonic(K: mesh.cell, psi, psi_hat, debug=False):

	if K.num_holes == 0:
		# simply-connected
		PSI, dPSI_dn_wgt = _get_anti_laplacian_simply_connected(
			K, psi, psi_hat
		)
	else:
		# multiply-connected
		PSI, dPSI_dn_wgt = _get_anti_laplacian_multiply_connected(
			K, psi, psi_hat, debug
		)

	return PSI, dPSI_dn_wgt

def _get_anti_laplacian_simply_connected(K: mesh.cell, psi, psi_hat):

	rho = _get_rho_from_grad(K, psi, - psi_hat)
	rho_hat = _get_rho_from_grad(K, psi_hat, psi)

	x1, x2 = K.get_boundary_points()

	PSI = 0.25 * (x1 * rho + x2 * rho_hat)

	PSI_x1 = 0.25 * (rho 	 + x1 * psi + x2 * psi_hat)
	PSI_x2 = 0.25 * (rho_hat + x2 * psi - x1 * psi_hat)

	dPSI_dn = K.dot_with_normal(PSI_x1, PSI_x2)
	dPSI_dn_wgt = K.multiply_by_dx_norm(dPSI_dn)

	return PSI, dPSI_dn_wgt

def _get_rho_from_grad(K, g1, g2):
	interval_length = 2 * np.pi * K.num_edges
	drho_dt = K.dot_with_tangent(g1, g2)
	drho_dt_wgt = K.multiply_by_dx_norm(drho_dt)
	return d2n.fft_deriv.fft_antiderivative(drho_dt_wgt, interval_length)

def _get_anti_laplacian_multiply_connected(K: mesh.cell, psi, psi_hat, debug):

	# single and double layer operators
	T1 = nystrom.single_layer.single_layer_mat(K)
	T2 = nystrom.double_layer.double_layer_mat(K)

	# get points on the boundary
	x1, x2 = K.get_boundary_points()

	# shift the origin to a point outside of K
	x1 -= K.ext_pt[0]
	x2 -= K.ext_pt[1]

	# square norm of boundary points in shifted coordinates
	x_norm_sq = x1 ** 2 + x2 ** 2

	# shifted boundary points dotted with unit tangent and normal
	xt = K.dot_with_tangent(x1, x2)
	xn = K.dot_with_normal(x1, x2)

	# rescale by square norm
	xt /= x_norm_sq
	xn /= x_norm_sq

	# mulitpy by derivative norm in prepartion for integration
	xt = K.multiply_by_dx_norm(xt)
	xn = K.multiply_by_dx_norm(xn)

	# array sizes
	N = K.num_pts

	# row sums of double layer operator
	T2_sum = np.sum(T2, 1)

	integrator_mat = d2n.harmconj.get_integrator(K)

	# block RHS
	b = np.zeros((2 * N,))
	b[:N] = T1 @ (+ xn * psi + xt * psi_hat)
	b[N:] = T1 @ (- xt * psi + xn * psi_hat)

	# define linear operator function
	def linop4antilap(x):

		# unpack
		rho = x[:N]
		rho_hat = x[N:]

		# allocate space
		y = np.zeros((2 * N,))

		# first block row
		y[:N] = d2n.harmconj.apply_T2(
			rho, T2, T2_sum, K.closest_vert_idx
		)
		y[:N] += T1 @ (+ xn * rho + xt * rho_hat)
		y[:N] += integrator_mat @ rho

		# second block row
		y[N:] = d2n.harmconj.apply_T2(
			rho_hat, T2, T2_sum, K.closest_vert_idx
		)
		y[N:] += T1 @ (- xt * rho + xn * rho_hat)
		y[N:] += integrator_mat @ rho_hat

		return y

	# define linear operator object
	A = LinearOperator(
		dtype = float,
		shape = (2 * N, 2 * N),
		matvec = linop4antilap
	)

	# solve Nystrom system
	x, flag = gmres(A, b, atol=1e-12, tol=1e-12)

	# unpack solution components
	rho = x[:N]
	rho_hat = x[N:]

	# weighted normal derivative of rho
	drho_dn_wgt = xn * (psi - rho) + xt * (psi_hat - rho_hat)

	# trace of anti-Laplacian
	PSI = 0.25 * x_norm_sq * rho

	# weighted normal derivative of anti-Laplacian
	dPSI_nd_wgt = x_norm_sq * (0.5 * xn * rho + 0.25 * drho_dn_wgt)

	# DEBUG
	if debug:
		print(f'gmres flag = {flag}')
		print('residual = %.4e'%(np.linalg.norm(A(x) - b)))
		d2n.harmconj.get_cond_num(A)
		print(integrator_mat @ rho)
		print(integrator_mat @ rho_hat)

	return PSI, dPSI_nd_wgt