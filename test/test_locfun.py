"""
	Run tests with
	python3 -m unittest
"""
import unittest
import sys
import os
sys.path.append(os.path.abspath('../puncturedfem'))

import puncturedfem as pf
import numpy as np

class TestLocalFunction(unittest.TestCase):

	def setUp(self) -> None:
		self.n = 32
		self.tol = 1e-6

	def tearDown(self) -> None:
		self.K = []
		self.v = []
		self.w = []

	def test_punctured_square(self):
		"""
		Sets up the mesh cell K and functions functions v,w as in
		examples/ex1a-square-hole.ipynb
		"""

		# define quadrature schemes
		q_trap = pf.quad(qtype='trap', n=self.n)
		q_kress = pf.quad(qtype='kress', n=self.n)

		# initialize list of edges as empty list
		edge_list = []

		# bottom: (0,0) to (1,0)
		e = pf.edge(etype='line', q=q_kress)
		e.join_points([0,0], [1,0])
		edge_list.append(e)

		# right: (1,0) to (1,1)
		e = pf.edge(etype='line', q=q_kress)
		e.join_points([1,0], [1,1])
		edge_list.append(e)

		# top: (1,1) to (0,1)
		e = pf.edge(etype='line', q=q_kress)
		e.join_points([1,1], [0,1])
		edge_list.append(e)

		# left: (0,1) to (0,0)
		e = pf.edge(etype='line', q=q_kress)
		e.join_points([0,1], [0,0])
		edge_list.append(e)

		# inner circular boundary
		e = pf.edge(etype='circle', q=q_trap)
		e.reverse_orientation()
		e.dialate(0.25)
		e.translate([0.5, 0.5])
		edge_list.append(e)

		# define mesh cell
		K = pf.cell(edge_list=edge_list)

		# get the coordinates of sampled boundary points
		x1, x2 = K.get_boundary_points()

		# set target value of logarithmic coefficient
		a_exact = 1

		# set point in hole interior
		xi = [0.5, 0.5]

		# define trace of v
		v_trace = np.exp(x1) * np.cos(x2) + \
			0.5 * a_exact * np.log((x1 - xi[0]) ** 2 + (x2 - xi[1]) ** 2) + \
			x1 ** 3 * x2 + x1 * x2 ** 3

		# create polynomial object
		v_laplacian = pf.polynomial([ [12.0, 1, 1] ])

		# create local function object
		v = pf.locfun(v_trace, v_laplacian)
		v.compute_all(K)

		# trace of w
		w_trace = (x1 - 0.5) / ((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2) + \
			x1 ** 3 + x1 * x2 ** 2

		# define a monomial term by specifying its multi-index and coefficient
		w_laplacian = pf.polynomial([ [8.0, 1, 0] ])

		# declare w as local function object
		w = pf.locfun(w_trace, w_laplacian)
		w.compute_all(K)

		# compute L^2 inner product
		l2_vw_exact = 1.39484950156676
		l2_vw_computed = v.compute_l2(w, K)
		l2_error = abs(l2_vw_computed - l2_vw_exact)

		# compare to exact values
		h1_vw_exact = 4.46481780319135
		h1_vw_computed = v.compute_h1(w, K)
		h1_error = abs(h1_vw_computed - h1_vw_exact)

		self.assertTrue(l2_error < self.tol)
		self.assertTrue(h1_error < self.tol)

	def build_pacman(self):
		pass

	def build_ghost(self):
		pass

if __name__ == '__main__':
	unittest.main()