import numpy as np

from ..quad import *
from ..mesh import *

def double_layer_mat(K: cell):
	"""
	Double layer potential operator matrix
	"""

	N = K.num_pts
	A = np.zeros((N,N))

	for i in range(K.num_edges):
		for j in range(K.num_edges):
			A[K.vert_idx[i] : K.vert_idx[i + 1],
			  K.vert_idx[j] : K.vert_idx[j + 1]] = \
				_double_layer_block(K.edge_list[i], K.edge_list[j])

	return A

def _double_layer_block(e: edge, f: edge):

	# allocate block
	B = np.zeros((e.num_pts - 1, f.num_pts - 1))

	# trapezoid step size
	h = 1 / (f.num_pts - 1)

	# check if edges are the same edge
	same_edge = e == f

	# adapt quadrature to accomodate both trapezoid and Kress
	if f.qtype[0:5] == 'kress':
		j_start = 1
	else:
		j_start = 0

	#
	for i in range(e.num_pts - 1):
		for j in range(j_start, f.num_pts - 1):

			if same_edge and i == j:
				B[i, i] = 0.5 * e.curvature[i]
			else:
				xy = e.x[:,i] - f.x[:,j]
				xy2 = np.dot(xy, xy)
				B[i, j] = np.dot(xy, f.unit_normal[:,j]) / xy2

			B[i, j] *= f.dx_norm[j] * h

	return B