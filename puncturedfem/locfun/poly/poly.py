import numpy as np
from .monomial import monomial
from .multi_index import multi_index_2

class polynomial:
	"""
	Treated as a list of monomial objects
	"""

	def __init__(self, coef_multidx_pairs=None):
		self.set(coef_multidx_pairs)

	def set(self, coef_multidx_pairs=None):
		self.monos = []
		if coef_multidx_pairs is None:
			return
		for triple in coef_multidx_pairs:
			if len(triple) != 3:
				raise Exception(
					'Every multi-index / coefficient pair must consist of' +
					'\n\t[0]:\tthe coefficient' +
					'\n\t[1]:\tthe exponent on x_1' +
					'\n\t[2]:\tthe exponent on x_2'
				)
			c = float(triple[0])
			alpha = multi_index_2([triple[1], triple[2]])
			m = monomial(alpha, c)
			self.add_monomial(m)
		self.consolidate()

	def copy(self):
		new = polynomial()
		new.add_monomials(self.monos)
		return new

	def add_monomial(self, m: monomial) -> None:
		"""
		Adds a monomial to the polynomial
		"""
		if not m.is_zero():
			self.monos.append(m)

	def add_monomials(self, monos: list[monomial]=None) -> None:
		for m in monos:
			self.add_monomial(m)
		self.consolidate()

	def remove_zeros(self):
		"""
		Removes terms with zero coefficients
		"""
		for i in range(len(self.monos), 0, -1):
			if self.monos[i - 1].is_zero():
				del self.monos[i - 1]

	def consolidate(self) -> None:
		"""
		Consolidates the coefficients of repeated indices
		"""
		N = len(self.monos)
		for i in range(N):
			for j in range(i + 1, N):
				if self.monos[i].alpha == self.monos[j].alpha:
					self.monos[i] += self.monos[j]
					self.monos[j] *= 0
		self.remove_zeros()
		self.sort()

	def sort(self) -> None:
		"""
		Sorts the monomials according to multi-index id

		(Using Insertion Sort since monomial list is assumed be be short)
		"""
		for i in range(len(self.monos)):
			j = i
			while j > 0 and self.monos[j - 1] > self.monos[j]:
				temp = self.monos[j - 1]
				self.monos[j - 1] = self.monos[j]
				self.monos[j] = temp
				j -= 1

	def add_monomial_with_id(self, coef: float, id: int) -> None:
		m = monomial()
		m.set_multidx_from_id(id)
		m.set_coef(coef)
		self.add_monomial(m)
		self.consolidate()

	def add_monomials_with_ids(self, coef_list: list[float],
			id_list: list[int]) -> None:
		if len(coef_list) != len(id_list):
			raise Exception(
				'number of coefficients and multi-indices must be equal')
		for i in range(len(coef_list)):
			self.add_monomial_with_id(coef_list[i], id_list[i])
		self.consolidate()

	def is_zero(self) -> bool:
		self.consolidate()
		return len(self.monos) == 0

	def set_to_zero(self) -> None:
		self.monos = []

	def eval(self, x: float, y: float) -> float:
		val = np.zeros(np.shape(x))
		for m in self.monos:
			val += m.eval(x, y)
		return val

	def pow(self, exponent: int):
		if not isinstance(exponent, int) or exponent < 0:
			raise ValueError(
				'Exponent must be nonnegative integer'
			)
		new = polynomial([[1.0, 0, 0]])
		for _ in range(exponent):
			new *= self
		return new

	def compose(self, q1, q2):
		new = polynomial()
		for m in self.monos:
			temp = q1.pow(m.alpha.x)
			temp *= q2.pow(m.alpha.y)
			# temp = q1.pow(m.alpha[0])
			# temp *= q2.pow(m.alpha[1])
			new += m.coef * temp
		return new

	def partial_deriv(self, var: str):
		new = polynomial()
		for m in self.monos:
			dm = m.partial_deriv(var)
			new.add_monomial(dm)
		return new

	def grad(self):
		gx = self.partial_deriv('x')
		gy = self.partial_deriv('y')
		return gx, gy

	def laplacian(self):
		gx, gy = self.grad()
		gxx = gx.partial_deriv('x')
		gyy = gy.partial_deriv('y')
		return gxx + gyy

	def anti_laplacian(self):
		new = polynomial()

		# define |(x, y)|^2 = x^2 + y^2
		p1 = polynomial()
		p1.add_monomials_with_ids([1, 1], [3, 5])

		# loop over monomial terms
		for m in self.monos:

			# anti-Laplacian of the monomial m
			N = m.alpha.order // 2

			# (x ^ 2 + y ^ 2) ^ {k + 1}
			pk = p1.copy()

			# Delta ^ k (x ^ 2 + y ^ 2) ^ alpha
			Lk = polynomial()
			Lk.add_monomial(m)

			# first term: k = 0
			scale = 0.25 / (1 + m.alpha.order)
			P_alpha = pk * Lk * scale

			# sum over k = 1 : N
			for k in range(1, N + 1):
				pk *= p1
				Lk = Lk.laplacian()
				scale *= -0.25 / ((k + 1) * (m.alpha.order + 1 - k))
				P_alpha += pk * Lk * scale

			# add c_alpha * P_alpha to new
			new += P_alpha

		return new

	def get_weighted_normal_derivative(self, K):
		x1, x2 = K.get_boundary_points()
		gx, gy = self.grad()
		gx_trace = gx.eval(x1, x2)
		gy_trace = gy.eval(x1, x2)
		nd = K.dot_with_normal(gx_trace, gy_trace)
		return K.multiply_by_dx_norm(nd)

	def __repr__(self) -> str:
		self.sort()
		if len(self.monos) == 0:
			return '+ (0) '
		msg = ''
		for m in self.monos:
			msg += m.__repr__()
		return msg

	def __eq__(self, other: object) -> bool:
		"""
		Tests equality between self and other
		"""
		if not isinstance(other, polynomial):
			raise TypeError('Cannot compare polynomial to non-polynomial')
		if len(self.monos) != len(other.monos):
			return False
		self.sort()
		other.sort()
		for i in range(len(self.monos)):
			if self.monos[i] != other.monos[i]:
				return False
		return True

	def __add__(self, other):
		"""
		Defines the addition operation self + other,
		where other is either another polynomial or a scalar
		"""
		if isinstance(other, polynomial):
			new = polynomial()
			for m in self.monos:
				new.add_monomial(m)
			for m in other.monos:
				new.add_monomial(m)
		elif isinstance(other, int) or isinstance(other, float):
			new = polynomial()
			for m in self.monos:
				new.add_monomial(m)
			constant = monomial()
			constant.set_multidx_from_id(0)
			constant.set_coef(other)
			new.add_monomial(constant)
		else:
			raise TypeError(
				'Addition with a polynomial must be with a scalar or' +
				' with another polynomial')
		new.consolidate()
		return new

	def __radd__(self, other):
		"""
		Defines the addition operator other + self,
		where other is either another polynomial or a scalar
		"""
		if isinstance(other, int) or isinstance(other, float):
			return self + other
		else:
			raise TypeError(
				'Addition with a polynomial must be with a scalar or' +
				' with another polynomial')

	def __iadd__(self, other):
		"""
		Defines the increment operation self += other
		where other is either another polynomial or a scalar
		"""
		if isinstance(other, polynomial):
			for m in other.monos:
				self.add_monomial(m)
			self.consolidate()
		elif isinstance(other, int) or isinstance(other, float):
			constant = monomial()
			constant.set_multidx_from_id(0)
			constant.set_coef(other)
			self.add_monomial(constant)
			self.consolidate()
		else:
			raise TypeError('Can only add polynomials to other polynomials' +
		   ' or scalars')
		return self

	def __mul__(self, other):
		"""
		Defines the multiplication operator self * other,
		where other is either another polynomial or a scalar
		"""
		if isinstance(other, polynomial):
			new = polynomial()
			for m in self.monos:
				for n in other.monos:
					new.add_monomial(m * n)
			new.consolidate()
			return new
		elif isinstance(other, int) or isinstance(other, float):
			new = polynomial()
			for m in self.monos:
				new.add_monomial(other * m.copy())
			return new
		else:
			raise TypeError(
				'Multiplication by polynomial must be by a scalar or' +
				' by another polynomial')

	def __rmul__(self, other):
		"""
		Defines the multiplication operator other * self,
		where other is either another polynomial or a scalar
		"""
		if isinstance(other, polynomial):
			return self * other
		elif isinstance(other, int) or isinstance(other, float):
			return self * other
		else:
			raise TypeError(
				'Multiplication by polynomial must be by a scalar or' +
				' by another polynomial')

	def __truediv__(self, other):
		"""
		Defines division by a scalar
		"""
		if isinstance(other, int) or isinstance(other, float):
			return self * (1 / other)
		else:
			raise TypeError('Division of a polynomial must be by a scalar')

	def __neg__(self):
		return -1 * self

	def __sub__(self, other):
		return self + (-other)