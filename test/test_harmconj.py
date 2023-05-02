"""
	Run tests with
	python3 -m unittest
"""
import unittest
import sys
import os
sys.path.append(os.path.abspath('../puncturedfem'))

import numpy as np
from puncturedfem import quad
from puncturedfem.mesh import edge, cell
from puncturedfem.locfun.d2n.harmconj import get_harmonic_conjugate

class TestHarmonicConjugate(unittest.TestCase):

	def setUp(self):
		self.n = 32
		self.q_trap = quad(n=self.n, qtype='trap')
		self.q_kress = quad(n=self.n, qtype='kress')
		self.num_conj_pairs = 5
		return None

	def run_all_tests_for_cell(self, K):
		for pair_id in range(self.num_conj_pairs):
			phi, phi_hat = self.harmonic_conjugate_pair(K, pair_id)
			phi_hat_computed, log_coeff = get_harmonic_conjugate(K, phi)
			max_phi_hat_error = \
				self.compute_harmonic_conjugate_error(phi_hat, phi_hat_computed)
			self.assertTrue(max_phi_hat_error < 1e-4)
			self.assertTrue(len(log_coeff) == K.num_holes)
			if K.num_holes > 0:
				max_log_coeff_error = np.max(np.abs(log_coeff))
				self.assertTrue(max_log_coeff_error < 1e-4)
		return None

	def harmonic_conjugate_pair(self, K: cell.cell, pair_id):
		if pair_id == 0:
			phi_fun = 		lambda x: 1
			phi_hat_fun = 	lambda x: 0
		elif pair_id == 1:
			phi_fun = 		lambda x: x[0]
			phi_hat_fun = 	lambda x: x[1]
		elif pair_id == 2:
			phi_fun = 		lambda x: x[1]
			phi_hat_fun = 	lambda x: - x[0]
		elif pair_id == 3:
			phi_fun = 		lambda x: x[0] ** 2 - x[1] ** 2
			phi_hat_fun = 	lambda x: 2 * x[0] * x[1]
		elif pair_id == 4:
			phi_fun = 		lambda x: np.exp(x[0]) * np.cos(x[1])
			phi_hat_fun = 	lambda x: np.exp(x[0]) * np.sin(x[1])
		else:
			raise Exception(f'Harmonic conjugate pair "{pair_id}" not found')
		phi = K.evaluate_function_on_boundary(phi_fun)
		phi_hat = K.evaluate_function_on_boundary(phi_hat_fun)
		return phi, phi_hat

	def compute_harmonic_conjugate_error(self, exact, computed):
		exact -= exact[0]
		computed -= computed[0]
		return np.max(np.abs(exact - computed))

	def test_simply_connected_smooth(self):
		"""
		Bean
		"""
		e =edge.edge(etype='bean', q=self.q_trap)
		K = cell.cell(edge_list=[e])
		self.run_all_tests_for_cell(K)
		return None

	def test_simply_connected_corner(self):
		"""
		Teardrop
		"""
		e = edge.edge(etype='teardrop', q=self.q_kress)
		K = cell.cell(edge_list=[e])
		self.run_all_tests_for_cell(K)
		return None

	def test_simply_connected_multiple_corners(self):
		"""
		Straight edge.edge, circular arc, sine wave
		"""
		edge_list = []

		e = edge.edge(etype='line', q=self.q_kress)
		e.join_points([0,0], [1,0])
		edge_list.append(e)

		e = edge.edge(etype='circular_arc', q=self.q_kress, H=-1)
		e.join_points([1,0], [1,1])
		edge_list.append(e)

		e = edge.edge(etype='sine_wave', q=self.q_kress, amp=-0.15, freq=3)
		e.join_points([1,1], [0,1])
		edge_list.append(e)

		e = edge.edge(etype='line', q=self.q_kress)
		e.join_points([0,1], [0,0])
		edge_list.append(e)

		K = cell.cell(edge_list=edge_list)
		self.run_all_tests_for_cell(K)
		return None

	def test_multiply_connected_smooth(self):
		"""
		Ellipse with circular and bean punctures
		"""
		edge_list = []

		e = edge.edge(etype='ellipse', q=self.q_trap, a=1.5, b=1)
		edge_list.append(e)

		e = edge.edge(etype='circle', q=self.q_trap)
		e.reverse_orientation()
		e.dialate(0.5)
		e.translate([-0.5,0])
		edge_list.append(e)

		e = edge.edge(etype='bean', q=self.q_trap)
		e.reverse_orientation()
		e.dialate(0.25)
		e.translate([0.456, 0.123])
		edge_list.append(e)

		K = cell.cell(edge_list=edge_list)
		self.run_all_tests_for_cell(K)
		return None

	def test_multiply_connected_corners(self):
		"""
		A super wacky domain
		"""
		edge_list = []

		e = edge.edge(etype='sine_wave', q=self.q_kress, amp=-0.05, freq=5)
		e.join_points([-2,-1], [2,-1])
		edge_list.append(e)

		e = edge.edge(etype='line', q=self.q_kress)
		e.join_points([2,-1], [2,1])
		edge_list.append(e)

		e = edge.edge(etype='sine_wave', q=self.q_kress, amp=-0.03, freq=5)
		e.join_points([2,1], [-2,1])
		edge_list.append(e)

		e = edge.edge(etype='circular_arc', q=self.q_kress, H=-0.4)
		e.join_points([-2,1], [-2,-1])
		edge_list.append(e)

		e = edge.edge(etype='teardrop', q=self.q_kress)
		e.reverse_orientation()
		e.dialate(0.25)
		e.translate([-1,0])
		edge_list.append(e)

		e = edge.edge(etype='bean', q=self.q_trap)
		e.reverse_orientation()
		e.dialate(0.25)
		e.translate([0, 0.423])
		edge_list.append(e)

		e = edge.edge(etype='line', q=self.q_kress)
		e.join_points([1,0.5], [1.5,0.5])
		edge_list.append(e)

		e = edge.edge(etype='circular_arc', q=self.q_kress, H=-0.8)
		e.join_points([1.5,0.5], [1.6,-0.4])
		edge_list.append(e)

		e = edge.edge(etype='circular_arc', q=self.q_kress, H=-0.2)
		e.join_points([1.6,-0.4], [1,0.5])
		edge_list.append(e)

		dialate_factor = 0.5
		for e in edge_list:
			e.dialate(dialate_factor)

		K = cell.cell(edge_list=edge_list)
		self.run_all_tests_for_cell(K)
		return None

if __name__ == '__main__':
	unittest.main()