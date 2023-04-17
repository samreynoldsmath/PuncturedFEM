import numpy as np
from matplotlib import path

from . import edge

class contour:

	__slots__ = (
		'id',
		'edge_list',
		'num_edges',
		'num_pts',
		'vert_idx',
	)

	def __init__(self, edge_list: list[edge.edge], id='') -> None:

		# optional identifier
		self.id = id

		# save edge list
		self.edge_list = edge_list

		# record number of edges
		self.num_edges = len(self.edge_list)

		# record the index of the starting point of each edge
		self._find_vert_idx_and_num_pts()

	### methods for evaluating traces

	def evaluate_function_on_contour(self, fun):
		"""
		Return fun(x) for each sampled point on contour
		"""
		y = np.zeros((self.num_pts,))
		for j in range(self.num_edges):
			y[self.vert_idx[j]:self.vert_idx[j+1]] \
				= self.edge_list[j].evaluate_function(fun, ignore_endpoint=True)
		return y

	def get_boundary_points(self):
		"""
		Returns the x1 and x2 coordinates of the boundary points
		"""
		x1_fun = lambda x: x[0]
		x1 = self.evaluate_function_on_contour(x1_fun)
		x2_fun = lambda x: x[1]
		x2 = self.evaluate_function_on_contour(x2_fun)
		return x1, x2

	def dot_with_tangent(self, v1, v2):
		"""
		Returns the dot product (v1, v2) * unit_tangent
		"""
		y = np.zeros((self.num_pts,))
		for j in range(self.num_edges):
			y[self.vert_idx[j]:self.vert_idx[j+1]] \
				= v1[self.vert_idx[j]:self.vert_idx[j+1]] \
					* self.edge_list[j].unit_tangent[0, :-1] \
				+ v2[self.vert_idx[j]:self.vert_idx[j+1]] \
					* self.edge_list[j].unit_tangent[1, :-1]
		return y

	def dot_with_normal(self, v1, v2):
		"""
		Returns the dot product (v1, v2) * unit_normal
		"""
		y = np.zeros((self.num_pts,))
		for j in range(self.num_edges):
			y[self.vert_idx[j]:self.vert_idx[j+1]] \
				= v1[self.vert_idx[j]:self.vert_idx[j+1]] \
					* self.edge_list[j].unit_normal[0, :-1] \
				+ v2[self.vert_idx[j]:self.vert_idx[j+1]] \
					* self.edge_list[j].unit_normal[1, :-1]
		return y

	def multiply_by_dx_norm(self, vals):
		"""
		Returns f multiplied against the norm of the derivative of
		the curve parameterization
		"""
		if len(vals) != self.num_pts:
			raise Exception('vals must be same length as boundary')
		y = np.zeros((self.num_pts,))
		for j in range(self.num_edges):
			y[self.vert_idx[j]:self.vert_idx[j+1]] \
				= vals[self.vert_idx[j]:self.vert_idx[j+1]] \
					* self.edge_list[j].dx_norm[:-1]
		return y

	def integrate_over_contour(self, vals):
		y = self.multiply_by_dx_norm(vals)
		return self.integrate_over_contour_preweighted(y)

	def integrate_over_contour_preweighted(self, vals):
		if len(vals) != self.num_pts:
			raise Exception('vals must be same length as boundary')
		y = np.zeros((self.num_pts,))
		for i in range(self.num_edges):
			h = 2 * np.pi / (self.edge_list[i].num_pts - 1)
			y[self.vert_idx[i]:self.vert_idx[i + 1]] = \
				h * vals[self.vert_idx[i]:self.vert_idx[i + 1]]
		return np.sum(y)

	def get_integrator(self):
		one = lambda x: 1
		A = self.evaluate_function_on_contour(one)
		A = self.multiply_by_dx_norm(A)
		for i in range(self.num_edges):
			h = 2 * np.pi / (self.edge_list[i].num_pts - 1)
			A[self.vert_idx[i]:self.vert_idx[i + 1]] *= h
		return A


	### methods for interior points

	def is_in_interior_contour(self, x, y):
		"""
		Returns True if the point (x,y) lies in the interior of the
		contour specified by edge_list, and returns false otherwise.

		Returns false if (x,y) lies on the boundary.

		If x,y are arrays of the same size, returns a boolean array
		of the same size.
		"""
		if x.shape != y.shape:
			raise Exception('x and y must have same size')

		is_inside = np.zeros(x.shape, dtype=bool)
		x1, x2 = self.get_boundary_points()
		p = path.Path(np.array([x1, x2]).transpose())

		if len(x.shape) == 1:
			M = x.shape[0]
			for i in range(M):
				is_inside[i] = p.contains_point([x[i], y[i]])
		elif len(x.shape) == 2:
			M, N = x.shape
			for i in range(M):
				for j in range(N):
					is_inside[i, j] = p.contains_point([x[i, j], y[i, j]])

		return is_inside


	def _find_vert_idx_and_num_pts(self):
		"""
		Get the index of the starting point of each edge,
		and record the total number of sampled points on the boundary
		"""
		self.num_pts = 0
		self.vert_idx = [0]
		for e in self.edge_list:
			self.num_pts += e.num_pts - 1
			self.vert_idx.append(self.num_pts)
		return None

	def _get_bounding_box(self):
		xmin = np.inf
		xmax = -np.inf
		ymin = np.inf
		ymax = -np.inf
		for e in self.edge_list:
			xmin = np.min([xmin, np.min(e.x[0, :])])
			xmax = np.max([xmax, np.max(e.x[0, :])])
			ymin = np.min([ymin, np.min(e.x[1, :])])
			ymax = np.max([ymax, np.max(e.x[1, :])])
		return xmin, xmax, ymin, ymax

	def _get_distance_to_boundary(self, x, y):
		"""
		Returns the minimum distance from (x,y) to a point on the boundary
		"""
		dist = np.inf
		for e in self.edge_list:
			dist2e = np.min((e.x[0, :] - x) ** 2 + (e.x[1, :] - y) ** 2)
			dist = np.min([dist, dist2e])
		return np.sqrt(dist)

	def _get_int_pt_simple_contour(self):
		"""
		Returns an interior point.

		Uses a brute force search. There is likely a more efficient way.
		"""

		# find region of interest
		xmin, xmax, ymin, ymax = self._get_bounding_box()

		# set minimum desired distance to the boundary
		TOL = 1e-2 * np.min([xmax - xmin, ymax - ymin])

		# search from M by N rectangular grid points
		M = 9
		N = 9

		d = 0.0

		while d < TOL:
			# set up grid
			x_coord = np.linspace(xmin, xmax, M)
			y_coord = np.linspace(ymin, ymax, N)
			x, y = np.meshgrid(x_coord, y_coord)

			# determine which points are in the interior
			is_inside = self.is_in_interior_contour(x, y)

			# for each interior point in grid, compute distance to the boundary
			dist = np.zeros(x.shape)
			for i in range(M):
				for j in range(N):
					if is_inside[i, j]:
						dist[i, j] = self._get_distance_to_boundary( \
							x[i, j], y[i, j]
						)

			# pick a point farthest from the boundary
			k = np.argmax(dist, keepdims=True)
			ii = k // M
			jj = k % M
			d = dist[ii, jj]

			# if the best candidate is too close to the boundary,
			# refine grid and search again
			M = 4 * (M // 2) + 1
			N = 4 * (N // 2) + 1

			if M * N > 1_000_000:
				raise Exception('Unable to locate an interior point')

		int_pt = np.zeros((2,))
		int_pt[0] = x[ii, jj]
		int_pt[1] = y[ii, jj]
		return int_pt

	def __repr__(self) -> str:
		msg = 'contour object\n'
		msg += f'num_edges: \t\t{self.num_edges}\n'
		msg += f'num_pts: \t\t{self.num_pts}\n'
		return msg