from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numpy import ndarray, zeros, nanmin, nanmax, inf
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from .globfunsp import global_function_space
from .bilinear_form import bilinear_form

RESET = "\033[0m"        # Reset all attributes
RED = "\033[31m"         # Red text
GREEN = "\033[32m"       # Green text
YELLOW = "\033[33m"      # Yellow text
BLUE = "\033[34m"        # Blue text
MAGENTA = "\033[35m"     # Magenta text
CYAN = "\033[36m"        # Cyan text

def print_color(s: str, color) -> None:
	print(color + s + RESET)

class solver:

	V: global_function_space
	a: bilinear_form
	glob_mat: csr_matrix
	glob_rhs: csr_matrix
	row_idx: list[int]
	col_idx: list[int]
	mat_vals: list[float]
	rhs_idx: list[int]
	rhs_vals: list[float]
	num_funs: int
	interior_values: list[list[ndarray]]

	def __init__(self, V: global_function_space, a: bilinear_form) -> None:
		self.set_global_function_space(V)
		self.set_bilinear_form(a)

	def __str__(self) -> str:
		s = 'solver:\n'
		s += '\tV: %s\n'%self.V
		s += '\ta: %s\n'%self.a
		s += '\tnum_funs: %d\n'%self.num_funs
		return s

	### SETTERS ################################################################

	def set_global_function_space(self, V: global_function_space) -> None:
		self.V = V

	def set_bilinear_form(self, a: bilinear_form) -> None:
		self.a = a

	### SOLVE LINEAR SYSTEM ####################################################

	def solve(self) -> ndarray:
		"""Solve linear system"""
		self.soln = spsolve(self.glob_mat, self.glob_rhs)

	def get_solution(self) -> ndarray:
		if self.soln is None:
			self.solve()
		return self.soln

	### ASSEMBLE GLOBAL SYSTEM #################################################

	def assemble(self, verbose: bool=True, processes: int=1) -> None:
		"""Assemble global system matrix"""
		self.build_values_and_indexes(verbose=verbose, processes=processes)
		self.find_num_funs()
		self.build_matrix_and_rhs()

	def build_values_and_indexes(self, verbose: bool=True,
		  processes: int=1) -> None:
		if processes == 1:
			self.build_values_and_indexes_sequential(verbose=verbose)
		elif processes > 1:
			self.build_values_and_indexes_parallel(verbose=verbose,
			  processes=processes)
		else:
			raise ValueError('processes must be a positive integer')

	def build_values_and_indexes_parallel(self, verbose: bool=True,
		  processes: int=1) -> None:
		raise NotImplementedError('Parallel assembly not yet implemented')

	def build_values_and_indexes_sequential(self, verbose: bool=True) -> None:

		if verbose:
			from tqdm import tqdm

		self.rhs_idx = []
		self.rhs_vals = []
		self.row_idx = []
		self.col_idx = []
		self.mat_vals = []

		self.interior_values = [[] for _ in range(self.V.T.num_cells)]

		# loop over cells
		for abs_cell_idx in range(self.V.T.num_cells):

			cell_idx = self.V.T.cell_idx_list[abs_cell_idx]

			if verbose:
				print_color(
					'Cell %6d / %6d'%(abs_cell_idx + 1, self.V.T.num_cells),
					GREEN)

			# build local function space
			V_K = self.V.build_local_function_space(cell_idx, verbose=verbose)

			# initialize interior values
			self.interior_values[abs_cell_idx] = \
				[None for _ in range(V_K.num_funs)]

			if verbose:
				print('Evaluating bilinear form and right-hand side...')

			# loop over local functions
			loc_basis = V_K.get_basis()

			if verbose:
				range_num_funs = tqdm(range(V_K.num_funs))
			else:
				range_num_funs = range(V_K.num_funs)

			for i in range_num_funs:

				v = loc_basis[i]

				# store interior values
				self.interior_values[abs_cell_idx][i] = v.int_vals

				# evaluate local right-hand side
				f_i = self.a.eval_rhs(v)

				# add to global right-hand side vector
				self.rhs_idx.append(v.id.glob_idx)
				self.rhs_vals.append(f_i)

				for j in range(i, V_K.num_funs):

					w = loc_basis[j]

					# evaluate local bilinear form
					a_ij = self.a.eval(v, w)

					# add to global stiffness matrix
					self.row_idx.append(v.id.glob_idx)
					self.col_idx.append(w.id.glob_idx)
					self.mat_vals.append(a_ij)

					# symmetry
					if j > i:
						self.row_idx.append(w.id.glob_idx)
						self.col_idx.append(v.id.glob_idx)
						self.mat_vals.append(a_ij)

		# TODO this is a hacky way to impose a zero Dirichlet BC
		for abs_cell_idx in range(self.V.T.num_cells):

			cell_idx = self.V.T.cell_idx_list[abs_cell_idx]

			for id in self.V.cell_dofs[abs_cell_idx]:

				if id.is_on_boundary:

					# zero rhs entry
					for k in range(len(self.rhs_idx)):
						if self.rhs_idx[k] == id.glob_idx:
							self.rhs_vals[k] = 0.0

					# zero mat row, except diagonal
					for k in range(len(self.row_idx)):
						if self.row_idx[k] == id.glob_idx:
							if self.col_idx[k] == id.glob_idx:
								self.mat_vals[k] = 1.0
							else:
								self.mat_vals[k] = 0.0

				# else:

				# 	# normalize row


	def find_num_funs(self) -> None:
		self.num_funs = self.V.num_funs
		self.check_sizes()

	def check_sizes(self) -> None:
		# rhs
		L = 1 + max(self.rhs_idx)
		if L > self.num_funs:
			raise Exception(f'L > self.num_funs ({L} > {self.num_funs})')
		# rows
		M = 1 + max(self.row_idx)
		if M > self.num_funs:
			raise Exception(f'M > self.num_funs ({M} > {self.num_funs})')
		# cols
		N = 1 + max(self.col_idx)
		if N > self.num_funs:
			raise Exception(f'N > self.num_funs ({N} > {self.num_funs})')

	def build_matrix_and_rhs(self) -> None:
		"""Build global system matrix and right-hand side vector"""
		self.glob_mat = csr_matrix(
			(self.mat_vals, (self.row_idx, self.col_idx)),
			shape=(self.num_funs, self.num_funs))
		self.glob_rhs = csr_matrix(
			(self.rhs_vals, (self.rhs_idx, [0] * len(self.rhs_idx))),
			shape=(self.num_funs, 1))

	### PLOT SOLUTION ##########################################################

	def plot_solution(self,
		   title='',
		   show_fig=True,
		   save_fig=False,
		   filename='solution.pdf',
		   fill=True) -> None:
		self.plot_linear_combo(self.soln,
			 title=title,
			 show_fig=show_fig,
			 save_fig=save_fig,
			 filename=filename,
			 fill=fill)

	def plot_linear_combo(self,
		       u: ndarray,
		       title='',
		       show_fig=True,
			   save_fig=False,
			   filename='solution.pdf',
			   fill=True) -> None:
		if not (show_fig or save_fig):
			return
		# compute linear combo on each cell, determine range of global values
		vals_arr = []
		vmin = inf
		vmax = -inf
		for cell_idx in self.V.T.cell_idx_list:
			coef = self.get_coef_on_cell(cell_idx, u)
			vals = self.compute_linear_combo_on_cell(cell_idx, coef)
			vals_arr.append(vals)
			vmin = min(vmin, nanmin(vals))
			vmax = max(vmax, nanmax(vals))

		# determine axes
		min_x = inf
		max_x = -inf
		min_y = inf
		max_y = -inf
		for e in self.V.T.edges:
			min_x = min(min_x, min(e.x[0, :]))
			max_x = max(max_x, max(e.x[0, :]))
			min_y = min(min_y, min(e.x[1, :]))
			max_y = max(max_y, max(e.x[1, :]))
		dx = max_x - min_x
		dy = max_y - min_y
		h = 4.0
		w = h * dx / dy

		# get figure object
		fig = plt.figure(figsize=(w, h))

		# plot mesh edges
		for e in self.V.T.edges:
			plt.plot(e.x[0, :], e.x[1, :], 'k')

		# plot interior values on each cell
		for cell_idx in self.V.T.cell_idx_list:
			vals = vals_arr[cell_idx]
			if vmax - vmin > 1e-6:
				K = self.V.T.get_cell(cell_idx)
				abs_cell_idx = self.V.T.get_abs_cell_idx(cell_idx)
				K.parameterize(self.V.quad_dict)
				if fill:
					plt.contourf(
						K.int_x1, K.int_x2,	vals_arr[abs_cell_idx],
						vmin=vmin,
						vmax=vmax,
						levels=32,
					)
				else:
					plt.contour(
						K.int_x1, K.int_x2,	vals_arr[abs_cell_idx],
						vmin=vmin,
						vmax=vmax,
						levels=32,
						colors='b',
					)
		if fill:
			sm = ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax))
			plt.colorbar(sm, fraction=0.046, pad=0.04)
		plt.axis('equal')
		plt.axis('off')
		plt.gca().set_aspect('equal')
		plt.subplots_adjust(
			left=0.0, right=1.0,
			bottom=0.0, top=1.0,
		    wspace=0.0, hspace=0.0)

		if len(title) > 0:
			plt.title(title)

		if save_fig:
			plt.savefig(filename)

		if show_fig:
			plt.show()

		plt.close(fig)

	def compute_linear_combo_on_cell(self, cell_idx, coef: ndarray) -> None:
		vals = 0.0
		abs_cell_idx = self.V.T.get_abs_cell_idx(cell_idx)
		int_vals = self.interior_values[abs_cell_idx]
		for i in range(len(int_vals)):
			vals += int_vals[i] * coef[i]
		return vals

	def get_coef_on_cell(self, cell_idx: int, u: ndarray) -> ndarray:
		abs_cell_idx = self.V.T.get_abs_cell_idx(cell_idx)
		ids = self.V.cell_dofs[abs_cell_idx]
		coef = zeros(len(ids))
		for i in range(len(ids)):
			coef[i] = u[ids[i].glob_idx]
		return coef