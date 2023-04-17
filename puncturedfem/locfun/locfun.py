from .. import d2n
from .. import antilap
from ..mesh.cell import cell
from ..poly.poly import polynomial

class locfun:
	"""
	Local function object (defined over single cell), consisting of:
	* Dirichlet trace values (given as array)
	* Laplacian (given as polynomial object)
	* Polynomial part (anti-Laplacian of Laplacian)
	* Polynomial part trace (array)
	* Conjugate trace (trace of harmonic conjugate of conjugable part)
	* Logarithmic coefficients (for multiply connected domains)
	* Weighted normal derivative of harmonic part
	* Trace of anti-Laplacian of harmonic part
	* Weighted normal derivative of anti-Laplacian of harmonic part
	"""

	__slots__ = (
		'trace', 			# array
		'lap',				# polynomial
		'poly_part', 		# polynomial
		'poly_part_trace', 	# array
		'poly_part_wnd', 	# array
		'conj_trace',		# array
		'log_coef',			# (small) array
		'harm_part_wnd',	# array
		'antilap_trace',	# array
		'antilap_wnd',		# array
	)

	def __init__(self, boundary_trace_values, laplacian_polynomial) -> None:
		self.set_trace_values(boundary_trace_values)
		self.set_laplacian_polynomial(laplacian_polynomial)

	def compute_all(self, K: cell) -> None:
		"""
		Computes all relevant data for reducing volumetric integrals
		to boundary integrals
		"""
		self.compute_polynomial_part()
		self.compute_polynomial_part_trace(K)
		self.compute_polynomial_part_weighted_normal_derivative(K)
		self.compute_harmonic_conjugate(K)
		self.compute_harmonic_weighted_normal_derivative(K)
		self.compute_anti_laplacian_harmonic_part(K)

	### Dirichlet trace ########################################################
	def set_trace_values(self, vals) -> None:
		self.trace = vals

	def get_trace_values(self):
		return self.trace

	### Laplacian (polynomial) #################################################
	def set_laplacian_polynomial(self, p: polynomial) -> None:
		self.lap = p

	def get_laplacian_polynomial(self):
		return self.lap

	### polynomial part (polynomial anti-Laplacian of Laplacian) ###############
	def set_polynomial_part(self, P_poly) -> None:
		self.poly_part = P_poly

	def get_polynomial_part(self):
		return self.poly_part

	def compute_polynomial_part(self) -> None:
		self.poly_part = self.lap.anti_laplacian()

	### polynomial part trace ##################################################
	def set_polynomial_part_trace(self, P_trace) -> None:
		self.poly_part_trace = P_trace

	def get_polynomial_part_trace(self):
		return self.poly_part_trace

	def compute_polynomial_part_trace(self, K: cell) -> None:
		x1, x2 = K.get_boundary_points()
		self.poly_part_trace = self.poly_part.eval(x1, x2)

	### polynomial part weighted normal derivative #############################
	def set_polynomial_part_weighted_normal_derivative(self, P_wnd) -> None:
		self.poly_part_wnd = P_wnd

	def get_polynomial_part_weighted_normal_derivative(self):
		return self.poly_part_wnd

	def compute_polynomial_part_weighted_normal_derivative(self, K: cell):
		x1, x2 = K.get_boundary_points()
		g1, g2 = self.poly_part.grad()
		P_nd = K.dot_with_normal(g1.eval(x1, x2), g2.eval(x1, x2))
		self.poly_part_wnd = K.multiply_by_dx_norm(P_nd)

	### harmonic conjugate #####################################################
	def set_harmonic_conjugate(self, hc_vals) -> None:
		self.conj_trace = hc_vals

	def get_harmonic_conjugate(self):
		return self.conj_trace

	def compute_harmonic_conjugate(self, K, debug=False) -> None:
		phi_trace = self.trace - self.poly_part_trace
		self.conj_trace, self.log_coef = \
			d2n.harmconj.get_harmonic_conjugate(K, phi_trace, debug=debug)

	### logarithmic coefficients ###############################################
	def set_logarithmic_coefficients(self, log_coef) -> None:
		self.log_coef = log_coef

	def get_logarithmic_coefficients(self):
		return self.log_coef

	# no compute method, this is handled by compute_harmonic_conjugate()

	### weighted normal derivative of harmonic part ############################
	def set_harmonic_weighted_normal_derivative(self, hc_wnd) -> None:
		self.harm_part_wnd = hc_wnd

	def get_harmonic_weighted_normal_derivative(self):
		return self.harm_part_wnd

	def compute_harmonic_weighted_normal_derivative(self, K) -> None:
		self.harm_part_wnd = \
		d2n.trace2tangential.get_weighted_tangential_derivative_from_trace(
			K, self.conj_trace)
		lam_x1, lam_x2 = d2n.log_terms.get_log_grad(K)
		lam_wnd = d2n.log_terms.get_dlam_dn_wgt(K, lam_x1, lam_x2)
		self.harm_part_wnd += lam_wnd @ self.log_coef

	### harmonic conjugable part psi ###########################################
	def get_conjugable_part(self, K: cell):
		lam = d2n.log_terms.get_log_trace(K)
		return self.trace - self.poly_part_trace - lam @ self.log_coef

	### anti-Laplacian #########################################################
	def set_anti_laplacian_harmonic_part(self, anti_laplacian_vals) -> None:
		self.antilap_trace = anti_laplacian_vals

	def get_anti_laplacian_harmonic_part(self):
		return self.antilap_trace

	def compute_anti_laplacian_harmonic_part(self, K: cell) -> None:
		psi = self.get_conjugable_part(K)
		self.antilap_trace, self.antilap_wnd = \
			antilap.antilap.get_anti_laplacian_harmonic( \
			K, psi=psi, psi_hat=self.conj_trace, a=self.log_coef)

	### H^1 semi-inner product #################################################
	def compute_h1(self, other, K):
		"""
		Returns the H^1 semi-inner product
			\int_K grad(self) * grad(other) dx
		"""

		# polynomial part
		Px, Py = self.poly_part.grad()
		Qx, Qy = other.poly_part.grad()
		gradP_gradQ = Px * Qx + Py * Qy
		val = gradP_gradQ.integrate_over_cell(K)

		# remaining terms
		integrand = other.trace * self.harm_part_wnd \
			+ self.poly_part_trace * other.harm_part_wnd
		val += K.integrate_over_boundary_preweighted(integrand)

		return val


	### L^2 inner product ######################################################
	def compute_l2(self, other, K: cell):
		"""
		Returns the L^2 inner product
			\int_K (self) * (other) dx
		"""

		x1, x2 = K.get_boundary_points()

		# P * Q
		PQ = self.poly_part * other.poly_part
		val = PQ.integrate_over_cell(K)

		# phi * psi
		integrand = (other.trace - other.poly_part_trace) * self.antilap_wnd \
			- self.antilap_trace * other.harm_part_wnd

		# phi * Q
		R = other.poly_part.anti_laplacian()
		R_trace = R.eval(x1, x2)
		R_wnd = R.get_weighted_normal_derivative(K)
		integrand += (self.trace - self.poly_part_trace) * R_wnd \
			- R_trace * self.harm_part_wnd

		# psi * P
		R = self.poly_part.anti_laplacian()
		R_trace = R.eval(x1, x2)
		R_wnd = R.get_weighted_normal_derivative(K)
		integrand += (other.trace - other.poly_part_trace) * R_wnd\
			- R_trace * other.harm_part_wnd

		# integrate over boundary
		val += K.integrate_over_boundary_preweighted(integrand)

		return val