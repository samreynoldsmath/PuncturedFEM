import numpy as np

from ..quad.quad import quad
from ..mesh.cell import cell
from ..mesh.edge import edge

def single_layer_mat(K: cell):
	"""
	Single layer potential operator matrix
	"""

	N = K.num_pts
	A = np.zeros((N,N))

	for j in range(K.num_edges):

		# Martensen quadrature
		nm = (K.edge_list[j].num_pts - 1) // 2
		qm = quad(qtype='mart', n=nm)

		for i in range(K.num_edges):

			# get block corresponding to edges e and f
			A[K.vert_idx[i] : K.vert_idx[i + 1],
			  K.vert_idx[j] : K.vert_idx[j + 1]] = \
				_single_layer_block(K.edge_list[i], K.edge_list[j], qm)

	return A

def _single_layer_block(e: edge, f: edge, qm: quad):
	"""
	Returns a block in the single layer matrix corresponding to
	x in edge e and y in edge f
	"""

	# allocate block
	B = np.zeros((e.num_pts - 1, f.num_pts - 1))

	# trapezoid weight: pi in integrand cancels
	h = -0.5 / (f.num_pts - 1)

	# check if two edges are the same
	same_edge = e == f

	# adapt quadrature to accomodate both trapezoid and Kress
	if f.qtype[0:5] == 'kress':
		j_start = 1
	else:
		j_start = 0

	if same_edge: # Kress and Martensen

		for i in range(e.num_pts - 1):
			for j in range(j_start,f.num_pts - 1):
				ij = abs(i-j)
				if ij == 0:
					B[i, i] =  2 * np.log(e.dx_norm[i])
				else:
					xy = e.x[:,i] - f.x[:,j]
					xy2 = np.dot(xy, xy)
					B[i, j] = np.log(xy2 / qm.t[ij])
				B[i, j] *= h
				B[i, j] += qm.wgt[ij]

	else: # different edges: Kress only

		for i in range(e.num_pts - 1):
			for j in range(j_start, f.num_pts - 1):
				xy = e.x[:,i] - f.x[:,j]
				xy2 = np.dot(xy, xy)
				B[i, j] = np.log(xy2) * h

	return B