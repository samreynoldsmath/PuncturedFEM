from numpy import zeros, ndarray

from puncturedfem.locfun.poly.poly import polynomial

from ...mesh.edge import edge
from ...mesh.cell import cell
from .poly import polynomial

class piecewise_polynomial:
	"""List of polynomials used to represent traces of vertex and edge funs"""

	polys: list[polynomial]
	num_polys: int
	id: int # used to associate function with an edge or vertex

	def __init__(self, num_polys: int=1,
	      polys: list[polynomial]=None,
		  id: int=0) -> None:
		self.set_id(id)
		self.set_num_polys(num_polys)
		self.set_polys(polys)

	def set_id(self, id: int) -> None:
		if not isinstance(id, int):
			raise TypeError('id must be an integer')
		if id < 0:
			raise ValueError('id must be nonnegative')
		self.id = id

	def set_num_polys(self, num_polys: int) -> None:
		if not isinstance(num_polys, int):
			raise TypeError('num_polys must be a integer')
		if num_polys < 1:
			raise ValueError('num_polys must be positve')
		self.num_polys = num_polys

	def set_polys(self, polys: list[polynomial]=None) -> None:
		if polys is None:
			self.polys = [polynomial() for _ in range(self.num_polys)]
		else:
			self.polys = polys

	def eval_on_edges(self, edges: list[edge]) -> ndarray:
		m = len(edges)
		if m != self.num_polys:
			raise Exception('Number of edges must match number of polynomials')
		vals_arr = []
		num_pts = 0
		for i in range(m):
			e = edges[i]
			num_pts += e.num_pts
			vals_arr.append(
				self.polys[i].eval(x=e.x[0, :], y=e.x[1, :])
			)
		vals = zeros((num_pts,))
		idx = 0
		for i in range(m):
			e = edges[i]
			vals[idx:idx + e.num_pts] = vals_arr[i]
		return vals

	def eval_on_cell_boundary(self, K: cell) -> ndarray:
		edges = K.get_edges()
		return self.eval_on_edges(edges)