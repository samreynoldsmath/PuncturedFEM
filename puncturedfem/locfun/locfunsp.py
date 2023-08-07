from ..mesh.cell import cell
from .edge_space import edge_space
from .locfun import locfun
from .locfun import polynomial
from .nystrom import nystrom_solver
from ..solver.globkey import global_key

class locfunspace:
	"""
	A collection of local functions (locfun objects) that form a basis of the
	local Poisson space V_p(K). The locfuns are partitioned into three types:
		vert_funs: vertex functions (harmonic, trace supported on two edges)
		edge_funs: edge functions (harmonic, trace supported on one edge)
		bubb_funs: bubble functions (polynomial Laplacian, zero trace)
	In the case where the mesh cell K has a vertex-free edge (that is, the edge
	is a simple closed contour, e.g. a circle), no vertex functions are
	associated with that edge.

	Edge spaces for all edges in the cell K must be computed first.
	"""

	deg: int
	num_vert_funs: int
	num_edge_funs: int
	num_bubb_funs: int
	num_funs: int
	vert_funs: list[locfun]
	edge_funs: list[locfun]
	bubb_funs: list[locfun]
	solver: nystrom_solver

	def __init__(self, K: cell, edge_spaces: list[edge_space], deg: int=1,
	      verbose: bool=True, processes: int=1) -> None:

		# set polynomial degree
		self.set_deg(deg)

		# set up nyström solver
		self.solver = nystrom_solver(K, verbose=verbose)

		# bubble functions: zero trace, polynomial Laplacian
		self.build_bubble_funs()

		# build vertex functions...')
		self.build_vert_funs(edge_spaces)

		# build edge functions
		self.build_edge_funs(edge_spaces)

		# count number of each type of function
		self.compute_num_funs()

		# compute all function metadata
		self.compute_all(verbose=verbose, processes=processes)

		# find interior values
		self.find_interior_values(verbose=verbose)

	def set_deg(self, deg: int) -> None:
		if not isinstance(deg, int):
			raise TypeError('deg must be an integer')
		if deg < 1:
			raise ValueError('deg must be a positive integer')
		self.deg = deg

	def find_interior_values(self, verbose: bool=True) -> None:
		if verbose:
			print('Finding interior values...')
			from tqdm import tqdm
			basis = tqdm(self.get_basis())
		else:
			basis = self.get_basis()
		for v in basis:
			v.compute_interior_values()

	### BUILD FUNCTIONS ########################################################

	def compute_num_funs(self) -> None:
		"""
		Sum the number of vertex, edge, and bubble functions
		"""
		self.num_vert_funs = len(self.vert_funs)
		self.num_edge_funs = len(self.edge_funs)
		self.num_bubb_funs = len(self.bubb_funs)
		self.num_funs = \
			self.num_vert_funs + \
			self.num_edge_funs + \
			self.num_bubb_funs

	def compute_all(self, verbose: bool=True, processes: int=1) -> None:
		"""
		Equivalent to running v.compute_all(K) for each locfun v
		"""
		if processes == 1:
			self.compute_all_sequential(verbose=verbose)
		elif processes > 1:
			self.compute_all_parallel(verbose=verbose, processes=processes)
		else:
			raise ValueError('processes must be a positive integer')

	def compute_all_sequential(self, verbose: bool=True) -> None:
		if verbose:
			print('Computing function metadata...')
			from tqdm import tqdm
			basis = tqdm(self.get_basis())
		else:
			basis = self.get_basis()
		for v in basis:
			v.compute_all()

	def compute_all_parallel(self, verbose: bool=True,
		  processes: int=1) -> None:
		raise NotImplementedError('Parallel computation not yet implemented')

	def build_bubble_funs(self) -> None:

		# bubble functions
		num_bubb = (self.deg * (self.deg - 1)) // 2
		self.bubb_funs = []
		for k in range(num_bubb):
			v_id = global_key(fun_type='bubb', bubb_space_idx=k)
			v = locfun(solver=self.solver, id=v_id)
			p = polynomial()
			p.add_monomial_with_id(coef=1.0, id=k)
			v.set_laplacian_polynomial(p)
			self.bubb_funs.append(v)

	def build_vert_funs(self, edge_spaces: list[edge_space]) -> None:
		"""Construct vertex functions from edge spaces"""

		# find all vertices on cell
		vert_idx_set = set()
		for c in self.solver.K.components:
			for e in c.edges:
				if not e.is_loop:
					vert_idx_set.add(e.anchor.id)
					vert_idx_set.add(e.endpnt.id)
		vert_ids: list[global_key] = []
		for vert_idx in vert_idx_set:
			vert_ids.append(global_key(fun_type='vert', vert_idx=vert_idx))

		# initialize list of vertex functions and set traces
		self.vert_funs = []
		for vert_id in vert_ids:
			v = locfun(solver=self.solver, id=vert_id)
			for j in range(len(edge_spaces)):
				b = edge_spaces[j]
				for k in range(b.num_vert_funs):
					if b.vert_fun_global_keys[k].vert_idx == vert_id.vert_idx:
						v.poly_trace.polys[j] = b.vert_fun_traces[k]
			self.vert_funs.append(v)

	def build_edge_funs(self, edge_spaces: list[edge_space]) -> None:
		"""Construct edge functions from edge spaces"""

		# initialize list of edge functions
		self.edge_funs = []

		# loop over edges on cell
		for b in edge_spaces:

			# locate edge within cell
			glob_edge_idx = b.e.id
			glob_edge_idx_list = [e.id for e in self.solver.K.get_edges()]
			edge_idx = glob_edge_idx_list.index(glob_edge_idx)

			# loop over edge functions
			for k in range(b.num_edge_funs):

				v_trace = b.edge_fun_traces[k]

				# create harmonic locfun
				v = locfun(solver=self.solver, id=b.edge_fun_global_keys[k])

				# set Dirichlet data
				v.poly_trace.polys[edge_idx] = v_trace

				# add to list of edge functions
				self.edge_funs.append(v)

	def get_basis(self) -> list[locfun]:
		"""Return list of all functions"""
		return self.vert_funs + self.edge_funs + self.bubb_funs
