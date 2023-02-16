"""
	Run tests with
	python3 -m unittest
"""
import unittest
import numpy as np
from .. import d2n

class TestFFTDerivative(unittest.TestCase):

	def setUp(self):
		self.n = 16
		self.a = 0 * np.pi
		self.b = 6 * np.pi
		self.L = self.b - self.a
		h = self.L / (2 * self.n)
		self.t = np.linspace(self.a, self.b - h, 2 * self.n)
		self.num_function_pairs = 5
		self.TOL = 1e-3

	def test_fft_deriv(self):
		for pair_id in range(self.num_function_pairs):
			x_fun, dx_fun = self.get_function_pair(pair_id)
			x_val = x_fun(self.t)
			dx_val = dx_fun(self.t)
			dx_computed = d2n.fft_deriv.fft_derivative(x_val, self.L)
			dx_error = np.abs(dx_val - dx_computed)
			max_dx_error = np.max(dx_error)
			self.assertTrue(max_dx_error < self.TOL)

	def test_fft_antideriv(self):
		for pair_id in range(self.num_function_pairs):
			x_fun, dx_fun = self.get_function_pair(pair_id)
			x_val = x_fun(self.t)
			dx_val = dx_fun(self.t)
			x_computed = d2n.fft_deriv.fft_antiderivative(dx_val, self.L)
			x_computed += x_val[0] - x_computed[0]
			x_error = np.abs(x_val - x_computed)
			max_x_error = np.max(x_error)
			self.assertTrue(max_x_error < self.TOL)

	def get_function_pair(self, pair_id):
		if pair_id == 0:
			x = lambda t: np.ones(np.shape(self.t))
			dx = lambda t: np.zeros(np.shape(self.t))
		elif pair_id == 1:
			x = lambda t: np.cos(t)
			dx = lambda t: - np.sin(t)
		elif pair_id == 2:
			x = lambda t: 2 * np.cos(t)
			dx = lambda t: - 2 * np.sin(t)
		elif pair_id == 3:
			x = lambda t: 5 * np.cos(3 * t) - (1 / 7) * np.sin(2  * t)
			dx = lambda t: - 15 * np.sin(3 * t) - (2 / 7) * np.cos(2 * t)
		elif pair_id == 4:
			x = lambda t: 16 * (t - self.a) ** 2 * (t - self.b) ** 2 \
				/ (self.b - self.a) ** 4
			dx = lambda t: 32 * (t - self.a) * (t - self.b) \
				* (2 * t - self.a - self.b) \
				/ (self.b - self.a) ** 4
		else:
			raise Exception(f'pair_id = {pair_id} not found')
		return x, dx

if __name__ == '__main__':
	unittest.main()