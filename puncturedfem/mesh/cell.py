from numpy import array, ndarray, zeros, sqrt, sum, shape, linspace, meshgrid
from numpy.linalg import norm
from .edge import edge, NotParameterizedError
from .closed_contour import closed_contour

class cell:
	"""List of edges parameterizing the boundary of a mesh cell"""

	id: int
	components: list[closed_contour]
	num_holes: int
	num_edges: int
	num_pts: int
	component_start_idx: list[int]
	closest_vert_idx: list[int]
	edge_orients: list[int]
	int_mesh_size: tuple[int, int]
	int_x1: ndarray
	int_x2: ndarray
	is_inside: ndarray

	def __init__(self, id: int, edges: list[edge]) -> None:
		self.set_id(id)
		self.find_edge_orientations(edges)
		self.components = []
		self.find_boundary_components(edges)
		self.find_num_edges()

	### MESH TOPOLOGY ##########################################################

	def set_id(self, id: int) -> None:
		if not isinstance(id, int):
			raise TypeError(
				f'id = {id} invalid, must be a positive integer'
			)
		if id < 0:
			raise ValueError(
				f'id = {id} invalid, must be a positive integer'
			)
		self.id = id

	def find_edge_orientations(self, edges: list[edge]) -> list[int]:
		self.edge_orients = []
		for e in edges:
			if self.id == e.pos_cell_idx:
				self.edge_orients.append(+1)
			elif self.id == e.neg_cell_idx:
				self.edge_orients.append(-1)
			else:
				self.edge_orients.append(0)

	### LOCAL EDGE MANAGMENT ###################################################

	def find_num_edges(self) -> int:
		self.num_edges = int(sum([c.num_edges for c in self.components]))

	def get_edges(self) -> list[edge]:
		edges = []
		for c in self.components:
			for e in c.edges:
				edges.append(e)
		return edges

	def get_edge_endpoint_incidence(self, edges: list[edge]):
		"""
		Returns incidence array: for each edge i, point to an edge j
		whose starting point is the terminal point of edge i

			edge i		  vertex 	edge j
			--->--->--->--- o --->--->--->---
		"""
		# if not self.is_parameterized():
		# 	raise NotParameterizedError('finding edge endpoint incidence')

		# form distance matrix between endpoints of edges
		num_edges = len(edges)
		distance = zeros((num_edges, num_edges))
		for i in range(num_edges):
			if self.edge_orients[i] == +1:
				a = edges[i].anchor
			elif self.edge_orients[i] == -1:
				a = edges[i].endpnt
			for j in range(num_edges):
				if self.edge_orients[j] == +1:
					b = edges[j].endpnt
				elif self.edge_orients[j] == -1:
					b = edges[j].anchor
				distance[i, j] = sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

		# mark edges as incident if distance between endpoints is zero
		TOL = 1e-6
		incidence_mat = zeros(distance.shape, dtype=int)
		for i in range(num_edges):
			for j in range(num_edges):
				if distance[i, j] < TOL:
					incidence_mat[i, j] = 1

		# check that each edge endpoint is incident to exactly one other edge
		row_sum = sum(incidence_mat, axis=0)
		rows_all_sum_to_one = norm(row_sum - 1) < TOL

		col_sum = sum(incidence_mat, axis=1)
		cols_all_sum_to_one = norm(col_sum - 1) < TOL

		if not (rows_all_sum_to_one and cols_all_sum_to_one):
			raise Exception(
				'Edge collection must be a union of ' +
				'disjoint simple closed contours'
			)

		# for each edge, return the index of the edge following it
		incidence = zeros((num_edges,), dtype=int)
		for i in range(num_edges):
			j = 0
			while incidence_mat[i, j] == 0:
				j += 1
			incidence[j] = i

		return incidence

	def find_boundary_components(self, edges: list[edge]) -> None:
		"""Finds the boundary components of the cell"""
		if not self.is_parameterized():
			raise NotParameterizedError('finding closed contours')

		contour_idx = []
		num_edges = len(edges)
		incidence = self.get_edge_endpoint_incidence(edges)

		is_marked_edge = zeros((num_edges,), dtype=bool)
		num_marked_edges = 0

		while num_marked_edges < num_edges:
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

			contour_idx.append(edges_on_contour)

		self.num_holes = -1 + len(contour_idx)

		for c_idx in contour_idx:
			edges_c = [edges[i] for i in c_idx]
			edge_orients_c = [self.edge_orients[i] for i in c_idx]
			self.components.append(closed_contour(
				cell_id=self.id,
				edges=edges_c,
				edge_orients=edge_orients_c))

	def find_hole_interior_points(self):
		"""
		Automatically find a point in the interior of each hole.

		Finds a point by creating a rectangular grid of points and
		eliminating those that are not in the interior. Among those
		that are in the interior, a point that lies a maximum distance
		from the boundary is chosen.
		"""
		raise NotImplementedError()

	### PARAMETERIZATON ########################################################
	def is_parameterized(self) -> bool:
		return all([c.is_parameterized() for c in self.components])

	def parameterize(self, quad_dict: dict) -> None:
		"""Parameterize each edge"""
		for c in self.components:
			c.parameterize(quad_dict)
		self.find_num_pts()
		self.find_outer_boundary()
		self.find_component_start_idx()
		self.find_closest_vert_idx()
		self.generate_interior_points()

	def deparameterize(self) -> None:
		for c in self.components:
			c.deparameterize()
		self.num_pts = 0
		self.component_start_idx = None

	def find_num_pts(self) -> None:
		"""Record the total number of sampled points on the boundary"""
		if not self.is_parameterized():
			raise NotParameterizedError('finding num_pts')
		self.num_pts = sum([c.num_pts for c in self.components])

	def find_component_start_idx(self) -> None:
		"""Find the index of sampled points corresponding to each component"""
		if not self.is_parameterized():
			raise NotParameterizedError('finding component_start_idx')
		self.component_start_idx = []
		idx = 0
		for c in self.components:
			self.component_start_idx.append(idx)
			idx += c.num_pts
		self.component_start_idx.append(idx)

	def find_outer_boundary(self):
		"""Find the outer boundary of the cell"""
		if not self.is_parameterized():
			raise NotParameterizedError('finding outer boundary')
		# find component that contains all other components
		outer_boundary_idx = 0
		for i in range(self.num_holes + 1):
			for j in range(i + 1, self.num_holes + 1):
				# check if contour j is contained in contour i
				x1, x2 = self.components[j].get_sampled_points()
				is_inside = self.components[i].is_in_interior_contour(x1, x2)
				if all(is_inside):
					outer_boundary_idx = i
		# swap contour_idx[0] and the outer boundary index
		temp = self.components[0]
		self.components[0] = self.components[outer_boundary_idx]
		self.components[outer_boundary_idx] = temp

	def find_closest_vert_idx(self) -> None:
		"""Find the closest vertex in the mesh to each sampled point"""
		if not self.is_parameterized():
			raise NotParameterizedError('finding closest_vert_idx')
		self.closest_vert_idx = zeros((self.num_pts,), dtype=int)
		for i in range(self.num_holes + 1):
			j = self.component_start_idx[i]
			jp1 = self.component_start_idx[i+1]
			self.closest_vert_idx[j:jp1] = self.components[i].closest_vert_idx

	### INTERIOR POINTS ########################################################

	def get_bounding_box(self) -> tuple[float, float, float, float]:
		"""Returns the bounding box of the cell"""
		if not self.is_parameterized():
			raise NotParameterizedError('getting bounding box')
		x1, x2 = self.get_boundary_points()
		xmin = min(x1)
		xmax = max(x1)
		ymin = min(x2)
		ymax = max(x2)
		return xmin, xmax, ymin, ymax

	def is_in_interior(self, x: array, y: array) -> ndarray[bool]:
		"""
		Returns a boolean array indicating whether each point is in the interior
		"""
		if not self.is_parameterized():
			raise NotParameterizedError('checking if points are in interior')
		is_in = zeros(shape(x), dtype=bool)
		# check if points are in outer boundary
		is_in = self.components[0].is_in_interior_contour(x, y)
		# check if points are in any of the holes
		for i in range(1, self.num_holes + 1):
			is_in = is_in & \
				~self.components[i].is_in_interior_contour(x, y)
		return is_in

	def get_distance_to_boundary(self, x: float, y: float) -> float:
		"""Returns the distance to the boundary at each point"""
		if not self.is_parameterized():
			raise NotParameterizedError('getting distance to boundary')
		dist = float('inf')
		for i in range(self.num_holes + 1):
			dist = min(dist, self.components[i].get_distance_to_boundary(x, y))
		return dist

	def generate_interior_points(self, rows=101, cols=101, tol=0.02):
		"""
		Returns (x, y, is_inside) where x,y are a meshgrid covering the
		cell K, and is_inside is a boolean array that is True for
		interior points
		"""

		self.int_mesh_size = (rows, cols)

		# find region of interest
		xmin, xmax, ymin, ymax = self.get_bounding_box()

		# set up grid
		x_coord = linspace(xmin, xmax, rows)
		y_coord = linspace(ymin, ymax, cols)
		self.int_x1, self.int_x2 = meshgrid(x_coord, y_coord)

		# determine which points are inside K
		self.is_inside = self.is_in_interior(self.int_x1, self.int_x2)

		# set minimum desired distance to the boundary
		TOL = tol * min([xmax - xmin, ymax - ymin])

		# ignore points too close to the boundary
		for i in range(rows):
			for j in range(cols):
				if self.is_inside[i, j]:
					d = self.get_distance_to_boundary(
						self.int_x1[i, j], self.int_x2[i, j])
					if d < TOL:
						self.is_inside[i, j] = False

	### FUNCTION EVALUATION ####################################################
	def evaluate_function_on_boundary(self, fun: callable) -> array:
		"""Return fun(x) for each sampled point on contour"""
		if not self.is_parameterized():
			raise NotParameterizedError('evaluating function on boundary')
		vals = zeros((self.num_pts,))
		for i in range(self.num_holes + 1):
			j = self.component_start_idx[i]
			jp1 = self.component_start_idx[i + 1]
			vals[j:jp1] = self.components[i].evaluate_function_on_contour(fun)
		return vals

	def get_boundary_points(self) -> tuple[array, array]:
		"""Returns the x1 and x2 coordinates of the boundary points"""
		if not self.is_parameterized():
			raise NotParameterizedError('getting boundary points')
		x1 = zeros((self.num_pts,))
		x2 = zeros((self.num_pts,))
		for i in range(self.num_holes + 1):
			j = self.component_start_idx[i]
			jp1 = self.component_start_idx[i + 1]
			x1[j:jp1], x2[j:jp1] = self.components[i].get_sampled_points()
		return x1, x2

	def dot_with_tangent(self, v1, v2) -> array:
		"""Returns the dot product (v1, v2) * unit_tangent"""
		if not self.is_parameterized():
			raise NotParameterizedError('dotting with tangent')
		if len(v1) != self.num_pts or len(v2) != self.num_pts:
			raise Exception('v1 and v2 must be same length as boundary')
		res = zeros((self.num_pts,))
		for i in range(self.num_holes + 1):
			j = self.component_start_idx[i]
			jp1 = self.component_start_idx[i + 1]
			res[j:jp1] = \
				self.components[i].dot_with_tangent(v1[j:jp1], v2[j:jp1])
		return res

	def dot_with_normal(self, v1, v2) -> array:
		"""Returns the dot product (v1, v2) * unit_normal"""
		if not self.is_parameterized():
			raise NotParameterizedError('dotting with normal')
		if len(v1) != self.num_pts or len(v2) != self.num_pts:
			raise Exception('v1 and v2 must be same length as boundary')
		res = zeros((self.num_pts,))
		for i in range(self.num_holes + 1):
			j = self.component_start_idx[i]
			jp1 = self.component_start_idx[i + 1]
			res[j:jp1] = \
				self.components[i].dot_with_normal(v1[j:jp1], v2[j:jp1])
		return res

	def multiply_by_dx_norm(self, vals) -> array:
		"""
		Returns f multiplied against the norm of the derivative of
		the curve parameterization
		"""
		if not self.is_parameterized():
			raise NotParameterizedError('multiplying by dx_norm')
		if len(vals) != self.num_pts:
			raise Exception('vals must be same length as boundary')
		vals_dx_norm = zeros((self.num_pts,))
		for i in range(self.num_holes + 1):
			j = self.component_start_idx[i]
			jp1 = self.component_start_idx[i + 1]
			vals_dx_norm[j:jp1] = \
				self.components[i].multiply_by_dx_norm(vals[j:jp1])
		return vals_dx_norm

	### INTEGRATION ############################################################
	def integrate_over_boundary(self, vals) -> float:
		"""Integrate vals over the boundary"""
		if not self.is_parameterized():
			raise NotParameterizedError('integrating over boundary')
		vals_dx_norm = self.multiply_by_dx_norm(vals)
		return self.integrate_over_boundary_preweighted(vals_dx_norm)

	def integrate_over_boundary_preweighted(self, vals_dx_norm) -> float:
		"""Integrate vals over the boundary without multiplying by dx_norm"""
		# if not self.is_parameterized():
		# 	raise NotParameterizedError('integrating over boundary')
		# if len(vals_dx_norm) != self.num_pts:
		# 	raise Exception('vals must be same length as boundary')
		# res = 0
		# for c, i in zip(self.components, range(self.num_holes + 1)):
		# 	j = self.component_start_idx[i]
		# 	jp1 = self.component_start_idx[i + 1]
		# 	res += c.integrate_over_closed_contour_preweighted(
		# 		vals_dx_norm[j:jp1])
		# return res

		# TODO: fix this hack
		# numpy.sum() is more stable, but this uses more memory

		from numpy import pi

		y = zeros((self.num_pts,))
		for i in range(self.num_holes + 1):
			c = self.components[i]
			h = 2 * pi * c.num_edges / c.num_pts
			j = self.component_start_idx[i]
			jp1 = self.component_start_idx[i + 1]
			y[j:jp1] = h * vals_dx_norm[j:jp1]
		return sum(y)