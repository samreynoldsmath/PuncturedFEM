from math import sqrt, floor

class multi_index_2:
	"""
	Integer multi-index with two components
	"""

	def __init__(self, alpha: list[int]=None) -> None:
		if alpha is None:
			alpha = [0, 0]
		self.set(alpha)

	def validate(self, alpha: list[int]):
		if not isinstance(alpha, list):
			raise TypeError('Multi-index must be list of two integers')
		if len(alpha) != 2:
			raise Exception('Multi-index is assumed to have 2 components')
		if not (isinstance(alpha[0], int) and isinstance(alpha[1], int)):
			raise TypeError('Multi-index must be list of two integers')
		if alpha[0] < 0 or alpha[1] < 0:
			raise ValueError('Components of multi-index must be nonnegative')

	def set(self, alpha: list[int]):
		self.validate(alpha)
		self.x = alpha[0]
		self.y = alpha[1]
		self.order = alpha[0] + alpha[1]
		self.id = alpha[1] + self.order * (self.order + 1) // 2

	def set_from_id(self, id: int):
		t = floor((sqrt(8 * id + 1) - 1) / 2)
		N = t * (t + 1) // 2
		alpha = []
		alpha.append(t - id + N)
		alpha.append(id - N)
		self.set(alpha)

	def copy(self):
		return multi_index_2([self.x, self.y])

	def __eq__(self, other) -> bool:
		if type(other) != multi_index_2:
			raise TypeError('Comparison of multi-index to object' +
			' of different type')
		return self.id == other.id

	def __add__(self, other):
		if isinstance(other, multi_index_2):
			beta = [self.x + other.x, self.y + other.y]
			return multi_index_2(beta)
		else:
			raise TypeError('Cannot add multi-index to different type')
