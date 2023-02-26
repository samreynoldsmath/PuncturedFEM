import numpy as np

from .edge import edge
from .contour import contour

class cell(contour):

	__slots__ = (
		'num_holes',
		'closest_vert_idx',
		'contour_idx',
		'hole_int_pts',
		'ext_pt'
	)

	def __init__(self, edge_list: list[edge]):

		# call initialization of contour object
		super().__init__(edge_list)

		# identify closed contours
		self._find_closed_contours()

		# identify outer boundary
		self._find_outer_boundary()

		# for each point on the boundary, find the nearest vertex
		# on the same contour
		self._find_closest_vertex_index()

		# find point in interior of each puncture automatically
		self._find_hole_int_pts()

		# find a point that is exterior to domain and not in a hole
		self._find_ext_pt()

	def integrate_over_boundary(self, vals):
		return self.integrate_over_contour(vals)

	def _find_closed_contours(self):
		"""
		for each edge, finds the index of which closed contour
		the edge belongs to, with 0 corresponding to the outer boundary
		"""

		self.contour_idx = []

		incidence = self._get_edge_endpoint_incidence()

		is_marked_edge = np.zeros((self.num_edges,), dtype=bool)
		num_marked_edges = 0

		while num_marked_edges < self.num_edges:
			edges_on_contour = []
			starting_edge = 0

			while is_marked_edge[starting_edge]:
				starting_edge += 1

			edges_on_contour.append(starting_edge)
			is_marked_edge[starting_edge] = True
			next_edge = incidence[starting_edge]

			while next_edge != starting_edge:
				edges_on_contour.append(next_edge)
				is_marked_edge[next_edge] = True
				next_edge = incidence[next_edge]

			num_marked_edges += len(edges_on_contour)

			self.contour_idx.append(edges_on_contour)

		self.num_holes = -1 + len(self.contour_idx)

	def _get_edge_endpoint_incidence(self):
		"""
		Returns incidence array: for each edge i, point to an edge j
		whose starting point is the terminal point of edge i

			edge i		  vertex 	edge j
			--->--->--->--- o --->--->--->---
		"""

		# form distance matrix between endpoints of edges
		distance = np.zeros((self.num_edges, self.num_edges))
		for i in range(self.num_edges):
			x = self.edge_list[i].x[:, 0]
			for j in range(self.num_edges):
				N = self.edge_list[j].num_pts
				y = self.edge_list[j].x[:, N - 1]
				distance[i, j] = np.linalg.norm(x - y)

		# mark edges as incident if distance between endpoints is zero
		TOL = 1e-6
		incidence_mat = np.zeros(distance.shape, dtype=int)
		for i in range(self.num_edges):
			for j in range(self.num_edges):
				if distance[i, j] < TOL:
					incidence_mat[i, j] = 1

		# check that each edge endpoint is incident to exactly one other edge
		row_sum = np.sum(incidence_mat, axis=0)
		rows_all_sum_to_one = np.linalg.norm(row_sum - 1) < TOL

		col_sum = np.sum(incidence_mat, axis=1)
		cols_all_sum_to_one = np.linalg.norm(col_sum - 1) < TOL

		if not (rows_all_sum_to_one and cols_all_sum_to_one):
			raise Exception('Edge collection must be a union of ' +
			'disjoint simple closed contours')

		# for each edge, return the index of the edge following it
		incidence = np.zeros((self.num_edges,), dtype=int)
		for i in range(self.num_edges):
			j = 0
			while incidence_mat[i, j] == 0:
				j += 1
			incidence[j] = i

		return incidence

	def _find_outer_boundary(self):
		outer_boundary_idx = -1
		for i in range(self.num_holes+1):
			if outer_boundary_idx < 0:
				edge_list_i = \
					[self.edge_list[k] for k in self.contour_idx[i]]
				ci = contour(edge_list=edge_list_i)
				for j in range(i, self.num_holes+1):
					edge_list_j = \
						[self.edge_list[k] for k in self.contour_idx[j]]
					cj = contour(edge_list=edge_list_j)
					# check if cj is contained in ci
					x1, x2 = cj.get_boundary_points()
					is_inside = ci.is_in_interior(x1, x2)
					if all(is_inside):
						outer_boundary_idx = i
		# swap contour_idx[0] and the outer boundary index
		temp = self.contour_idx[0]
		self.contour_idx[0] = self.contour_idx[outer_boundary_idx]
		self.contour_idx[outer_boundary_idx] = temp

	def _find_closest_vertex_index(self):

		# get midpoint indices
		mid_idx = np.zeros((self.num_edges,), dtype=int)
		for i in range(self.num_edges):
			n = self.edge_list[i].num_pts // 2 # 2n points per edge
			mid_idx[i] = self.vert_idx[i] + n

		# on first half of an edge, the closest vertex is the starting
		# point on that edge
		self.closest_vert_idx = np.zeros((self.num_pts,), dtype=int)
		for i in range(self.num_edges):
			self.closest_vert_idx[self.vert_idx[i]:mid_idx[i]] \
				= self.vert_idx[i]

		# on the second half of an edge, the closest vertex is the
		# starting point of the next edge on the same closed contour
		for c in self.contour_idx:
			m = len(c) # number of edges on this contour
			for k in range(m):
				i = c[k] # current edge
				j = c[(k + 1) % m] # next edge on contour
				n = self.edge_list[i].num_pts // 2 # 2n points per edge
				self.closest_vert_idx[mid_idx[i]:(mid_idx[i] + n)] \
					= self.vert_idx[j]

	def _find_hole_int_pts(self):
		"""
		Automatically find a point in the interior of each hole

		Finds a point by creating a rectangular grid of points and
		eliminating those that are not in the interior. Among those
		that are in the interior, a point that lies a maximum distance
		from the boundary is chosen.
		"""
		self.hole_int_pts = np.zeros((2, self.num_holes))
		for j in range(self.num_holes):
			c = contour(edge_list=[
				self.edge_list[i] for i in self.contour_idx[j + 1]
			])
			self.hole_int_pts[:, j] = c._get_int_pt_simple_contour()

	def _find_ext_pt(self):
		"""
		Find point to center origin such that cell lies strictly in the
		fist quadrant
		"""
		x1, x2 = self.get_boundary_points()
		x1_min = np.min(x1)
		x2_min = np.min(x2)
		self.ext_pt = np.array([x1_min, x2_min])

	def __repr__(self) -> str:
		msg = ''
		msg += f'num_edges: \t\t{self.num_edges}\n'
		msg += f'num_holes: \t\t{self.num_holes}\n'
		msg += f'num_pts: \t\t{self.num_pts}\n'
		msg += f'contours: \t\t{self.contour_idx}\n'
		msg += f'ext_pt: \t\t[{self.ext_pt[0]}, {self.ext_pt[1]}]\n'
		if self.num_holes > 0:
			msg += f'hole_int_pts (x): \t{self.hole_int_pts[0, :]}\n'
			msg += f'hole_int_pts (y): \t{self.hole_int_pts[1, :]}\n'
		return msg