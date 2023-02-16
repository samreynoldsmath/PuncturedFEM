import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from .. import mesh
from .. import nystrom

from . import log_terms
from . import trace2tangential

def get_harmonic_conjugate(K: mesh.cell, phi, debug=False):

	dphi_ds = trace2tangential.get_weighted_tangential_derivative_from_trace(K, phi)

	if K.num_holes == 0:
		# simply-connected
		psi_hat = get_harmonic_conjugate_simply_connected(
			K, dphi_ds, debug
		)
		return psi_hat, []
	else:
		# multiply-connected
		psi_hat, a = get_harmonic_conjugate_multiply_connected(
			K, phi, dphi_ds, debug
		)
		return psi_hat, a

def get_harmonic_conjugate_simply_connected(K: mesh.cell, dphi_ds, debug=False):

	# single and double layer operators
	T1 = nystrom.single_layer.single_layer_mat(K)
	T2 = nystrom.double_layer.double_layer_mat(K)

	# RHS
	b = - T1 @ dphi_ds

	T2_sum = np.sum(T2, 1)

	#
	integrator_mat = get_integrator(K)

	# total number of points on the boundary
	N = K.num_pts

	# define linear operator functionality
	def linop4harmconj(psi_hat):
		y = apply_T2(psi_hat, T2, T2_sum, K.closest_vert_idx)
		y += integrator_mat @ psi_hat
		return y

	# define linear operator object
	A = LinearOperator(
		dtype = float,
		shape = (N, N),
		matvec = linop4harmconj
	)

	# solve Nystrom system
	psi_hat, flag = gmres(A, b, atol=1e-12, tol=1e-12)

	# DEBUG
	if debug:
		print('gmres flag: get_harmonic_conjugate_simply_connected = ',flag)
		print(f'Condition number = {get_cond_num(A)}')

	return psi_hat

def get_harmonic_conjugate_multiply_connected(
	K: mesh.cell, phi, dphi_ds, debug=False):

	# get single and double layer operator matrices
	T1 = nystrom.single_layer.single_layer_mat(K)
	T2 = nystrom.double_layer.double_layer_mat(K)

	# traces and gradients of logarithmic corrections
	lam_trace = log_terms.get_log_trace(K)
	lam_x1_trace, lam_x2_trace = log_terms.get_log_grad(K)

	# tangential and normal derivatives of logarithmic terms
	dlam_dt_wgt = log_terms.get_dlam_dt_wgt(K, lam_x1_trace, lam_x2_trace)
	dlam_dn_wgt = log_terms.get_dlam_dn_wgt(K, lam_x1_trace, lam_x2_trace)

	# single layer operator applied to tangential derviatives of log terms
	T1_dlam_dt = T1 @ dlam_dt_wgt

	#
	Sn = get_Su(K, dlam_dn_wgt)
	St = get_Su(K, dlam_dt_wgt)

	# H1 seminorms of logarithmic terms
	Sn_lam = Sn @ lam_trace

	#
	integrator_mat = get_integrator(K)

	T2_sum = np.sum(T2, 1)

	# array sizes
	N = K.num_pts
	m = K.num_holes

	# block RHS
	b = np.zeros((N + m,))
	b[:N] = - T1 @ dphi_ds
	b[N:] = Sn @ phi

	def linop4harmconj(x):
		psi_hat = x[:N]
		a = x[N:]
		y = np.zeros((N + m,))
		y[:N] = apply_T2(psi_hat, T2, T2_sum, K.closest_vert_idx)
		y[:N] += integrator_mat @ psi_hat
		y[:N] -= T1_dlam_dt @ a
		y[N:] = - St @ psi_hat + Sn_lam @ a
		return y

	# define linear operator
	A = LinearOperator(
		dtype = float,
		shape = (N + m, N + m),
		matvec = linop4harmconj
	)

	# solve Nystrom system
	x, flag = gmres(A, b, atol=1e-12, tol=1e-12)
	psi_hat = x[:N]
	a = x[N:]

	# DEBUG
	if debug:
		print(f'gmres flag: get_harmonic_conjugate_multiply_connected = {flag}')
		print(f'Condition number = {get_cond_num(A)}')

	return psi_hat, a

def apply_T2(psi_hat, T2, T2_sum, closest_vert_idx):
	corner_values = psi_hat[closest_vert_idx]
	return 0.5 * (psi_hat - corner_values) \
		+ T2 @ psi_hat \
		- corner_values * T2_sum

def get_integrator(K):
	one = lambda x: 1
	A = K.evaluate_function_on_boundary(one)
	A = K.multiply_by_dx_norm(A)
	for i in range(K.num_edges):
		h = 2 * np.pi / (K.edge_list[i].num_pts - 1)
		A[K.vert_idx[i]:K.vert_idx[i + 1]] *= h
	return A

def get_integrator_on_outer(K):
	one = lambda x: 1
	A = K.evaluate_function_on_boundary(one)
	A = K.multiply_by_dx_norm(A)
	c = K.contour_idx[0]
	for i in c:
		h = 2 * np.pi / (K.edge_list[i].num_pts - 1)
		A[K.vert_idx[i]:K.vert_idx[i + 1]] *= h
	return A

def get_integrator_unweigted(K):
	one = lambda x: 1
	A = K.evaluate_function_on_boundary(one)
	# A = K.multiply_by_dx_norm(A)
	for i in range(K.num_edges):
		h = 2 * np.pi / (K.edge_list[i].num_pts - 1)
		A[K.vert_idx[i]:K.vert_idx[i + 1]] *= h
	return A

def get_Su(K, dlam_du_wgt):
	Su = np.zeros((K.num_holes, K.num_pts))
	Su[:,:] = np.transpose(dlam_du_wgt)
	for i in range(K.num_edges):
		h = 2 * np.pi / (K.edge_list[i].num_pts - 1)
		Su[:, K.vert_idx[i]:K.vert_idx[i + 1]] *= h
	return Su

def get_cond_num(A):
	"""
	FOR DEBUGGING
	"""
	n = np.shape(A)[0]
	I = np.eye(n)
	Amat = np.zeros((n, n))
	for j in range(n):
		Amat[:, j] = A(I[:, j])

	Amat = (Amat[:-2,:-2])
	n -= 2

	cond = np.linalg.cond(Amat)
	print('condition number = %.4e'%(cond))

	r = np.linalg.matrix_rank(Amat)
	print(f'rank = {r}')
	print(f'nullity = {n - r}')

	u, s, vh = np.linalg.svd((Amat))

	print(s[(n-10):n])

	import matplotlib.pyplot as plt

	plt.figure()
	plt.semilogy(s, 'k.')
	plt.title('singular values')
	plt.grid('on')

	plt.figure()
	plt.title('spanning set of the nullspace')
	leg = []
	for k in range(n):
		if s[k] < 1e-6:
			leg.append('%.4e'%(s[k]))
			plt.plot(vh[k,:])
	plt.legend(leg)

	u, s, vh = np.linalg.svd(np.transpose(Amat))
	plt.figure()
	plt.title('spanning set of the nullspace of transpose')
	leg = []
	for k in range(n):
		if s[k] < 1e-6:
			leg.append('%.4e'%(s[k]))
			plt.plot(vh[k,:])
	plt.legend(leg)

	plt.show()


	return