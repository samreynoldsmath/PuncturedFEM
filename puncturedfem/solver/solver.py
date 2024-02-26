"""
solver.py
=========

Module containing the Solver class, which is a convenience class for solving
the global linear system.
"""

from numpy import ndarray, shape, zeros
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from ..util.print_color import Color, print_color
from .bilinear_form import BilinearForm
from .globfunsp import GlobalFunctionSpace


class Solver:
    """
    Convenience class for solving the global linear system.

    Attributes
    ----------
    V : GlobalFunctionSpace
        Global function space
    a : BilinearForm
        Bilinear form
    glob_mat : csr_matrix
        Global system matrix
    glob_rhs : csr_matrix
        Global right-hand side vector
    row_idx : list[int]
        List of row indices
    col_idx : list[int]
        List of column indices
    mat_vals : list[float]
        List of matrix values
    rhs_idx : list[int]
        List of right-hand side indices
    rhs_vals : list[float]
        List of right-hand side values
    num_funs : int
        Number of global functions
    interior_values : list[list[ndarray]]
        List of interior values on each MeshCell
    soln : ndarray
        Solution vector
    """

    V: GlobalFunctionSpace
    a: BilinearForm
    stiff_mat: csr_matrix
    mass_mat: csr_matrix
    glob_mat: csr_matrix
    glob_rhs: csr_matrix
    row_idx: list[int]
    col_idx: list[int]
    mat_vals: list[float]
    stiff_vals: list[float]
    mass_vals: list[float]
    rhs_idx: list[int]
    rhs_vals: list[float]
    num_funs: int
    interior_values: list[list[ndarray]]
    soln: ndarray

    def __init__(self, V: GlobalFunctionSpace, a: BilinearForm) -> None:
        """
        Constructor for Solver class.

        Parameters
        ----------
        V : GlobalFunctionSpace
            Global function space
        a : BilinearForm
            Bilinear form
        """
        self.set_global_function_space(V)
        self.set_bilinear_form(a)

    def __str__(self) -> str:
        """
        Return string representation.
        """
        s = "Solver:\n"
        s += f"\tV: {self.V}\n"
        s += f"\ta: {self.a}\n"
        s += f"\tnum_funs: {self.num_funs}\n"
        return s

    # SETTERS ################################################################

    def set_global_function_space(self, V: GlobalFunctionSpace) -> None:
        """
        Set the global function space.
        """
        if not isinstance(V, GlobalFunctionSpace):
            raise TypeError("V must be a GlobalFunctionSpace")
        self.V = V

    def set_bilinear_form(self, a: BilinearForm) -> None:
        """
        Set the bilinear form.
        """
        if not isinstance(a, BilinearForm):
            raise TypeError("a must be a BilinearForm")
        self.a = a

    # SOLVE LINEAR SYSTEM ####################################################

    def solve(self) -> None:
        """Solve linear system"""
        self.soln = spsolve(self.glob_mat, self.glob_rhs)

    def get_solution(self) -> ndarray:
        """Return solution vector"""
        if self.soln is None:
            self.solve()
        return self.soln

    # ASSEMBLE GLOBAL SYSTEM #################################################

    def assemble(
        self,
        verbose: bool = True,
        processes: int = 1,
        compute_interior_values: bool = True,
    ) -> None:
        """Assemble global system matrix"""
        self.build_values_and_indexes(
            verbose=verbose,
            processes=processes,
            compute_interior_values=compute_interior_values,
        )
        self.find_num_funs()
        self.build_matrix_and_rhs()
        self.build_stiffness_matrix()
        self.build_mass_matrix()

    def build_values_and_indexes(
        self,
        verbose: bool = True,
        processes: int = 1,
        compute_interior_values: bool = True,
    ) -> None:
        """Build values and indexes"""
        if processes == 1:
            self.build_values_and_indexes_sequential(
                verbose=verbose, compute_interior_values=compute_interior_values
            )
        elif processes > 1:
            self.build_values_and_indexes_parallel(
                verbose=verbose,
                processes=processes,
                compute_interior_values=compute_interior_values,
            )
        else:
            raise ValueError("processes must be a positive integer")

    def build_values_and_indexes_parallel(
        self,
        verbose: bool = True,
        processes: int = 1,
        compute_interior_values: bool = True,
    ) -> None:
        """
        Build values and indexes in parallel.
        """
        raise NotImplementedError("Parallel assembly not yet implemented")

    def build_values_and_indexes_sequential(
        self, verbose: bool = True, compute_interior_values: bool = True
    ) -> None:
        """
        Build values and indexes sequentially.
        """

        self.rhs_idx = []
        self.rhs_vals = []
        self.row_idx = []
        self.col_idx = []
        self.mat_vals = []
        self.stiff_vals = []
        self.mass_vals = []

        self.interior_values = [[] for _ in range(self.V.T.num_cells)]

        # loop over MeshCells
        for abs_cell_idx in range(self.V.T.num_cells):
            cell_idx = self.V.T.cell_idx_list[abs_cell_idx]

            if verbose:
                print_color(
                    f"Cell {abs_cell_idx + 1:6} / {self.V.T.num_cells:6}",
                    Color.GREEN,
                )

            # build local function space
            V_K = self.V.build_local_function_space(
                cell_idx,
                verbose=verbose,
                compute_interior_values=compute_interior_values,
            )

            # initialize interior values
            if compute_interior_values:
                self.interior_values[abs_cell_idx] = [
                    zeros((0,)) for _ in range(V_K.num_funs)
                ]

            if verbose:
                print("Evaluating bilinear form and right-hand side...")

            # loop over local functions
            loc_basis = V_K.get_basis()

            range_num_funs: range | tqdm[int]
            if verbose:
                range_num_funs = tqdm(range(V_K.num_funs))
            else:
                range_num_funs = range(V_K.num_funs)

            for i in range_num_funs:
                v = loc_basis[i]

                # store interior values
                if compute_interior_values:
                    self.interior_values[abs_cell_idx][i] = v.int_vals

                # evaluate local right-hand side
                f_i = self.a.eval_rhs(v)

                # add to global right-hand side vector
                self.rhs_idx.append(v.key.glob_idx)
                self.rhs_vals.append(f_i)

                for j in range(i, V_K.num_funs):
                    w = loc_basis[j]

                    # evaluate local bilinear form
                    h1_ij = self.a.eval_h1(v, w)
                    l2_ij = self.a.eval_l2(v, w)
                    a_ij = self.a.eval_with_h1_and_l2(h1_ij, l2_ij)

                    # add to matrices
                    self.row_idx.append(v.key.glob_idx)
                    self.col_idx.append(w.key.glob_idx)
                    self.mat_vals.append(a_ij)
                    self.stiff_vals.append(h1_ij)
                    self.mass_vals.append(l2_ij)

                    # symmetry
                    if j > i:
                        self.row_idx.append(w.key.glob_idx)
                        self.col_idx.append(v.key.glob_idx)
                        self.mat_vals.append(a_ij)
                        self.stiff_vals.append(h1_ij)
                        self.mass_vals.append(l2_ij)

        # TODO this is a hacky way to impose a zero Dirichlet BC
        for abs_cell_idx in range(self.V.T.num_cells):
            cell_idx = self.V.T.cell_idx_list[abs_cell_idx]

            for key in self.V.cell_dofs[abs_cell_idx]:
                if key.is_on_boundary:
                    # zero rhs entry
                    for k, idx in enumerate(self.rhs_idx):
                        if idx == key.glob_idx:
                            self.rhs_vals[k] = 0.0
                    # zero mat row, except diagonal
                    for k, idx in enumerate(self.row_idx):
                        if idx == key.glob_idx:
                            if self.col_idx[k] == key.glob_idx:
                                self.mat_vals[k] = 1.0
                            else:
                                self.mat_vals[k] = 0.0

    def find_num_funs(self) -> None:
        """
        Find the number of global functions and run size checks.
        """
        self.num_funs = self.V.num_funs
        self.check_sizes()

    def check_sizes(self) -> None:
        """
        Check that the sizes of the global system matrix and right-hand side
        vector are correct.
        """

        # rhs
        L = 1 + max(self.rhs_idx)
        if L > self.num_funs:
            raise ValueError(f"L > self.num_funs ({L} > {self.num_funs})")
        # rows
        M = 1 + max(self.row_idx)
        if M > self.num_funs:
            raise ValueError(f"M > self.num_funs ({M} > {self.num_funs})")
        # cols
        N = 1 + max(self.col_idx)
        if N > self.num_funs:
            raise ValueError(f"N > self.num_funs ({N} > {self.num_funs})")

    def build_matrix_and_rhs(self) -> None:
        """Build global system matrix and right-hand side vector"""
        self.glob_mat = csr_matrix(
            (self.mat_vals, (self.row_idx, self.col_idx)),
            shape=(self.num_funs, self.num_funs),
        )
        self.glob_rhs = csr_matrix(
            (self.rhs_vals, (self.rhs_idx, [0] * len(self.rhs_idx))),
            shape=(self.num_funs, 1),
        )

    def build_stiffness_matrix(self) -> None:
        """Build stiffness matrix"""
        self.stiff_mat = csr_matrix(
            (self.stiff_vals, (self.row_idx, self.col_idx)),
            shape=(self.num_funs, self.num_funs),
        )

    def build_mass_matrix(self) -> None:
        """Build mass matrix"""
        self.mass_mat = csr_matrix(
            (self.mass_vals, (self.row_idx, self.col_idx)),
            shape=(self.num_funs, self.num_funs),
        )

    # COMPUTE LINEAR COMBINATION #############################################

    def compute_linear_combo_on_mesh(
        self, cell_idx: int, coef: ndarray
    ) -> ndarray:
        """
        Compute a linear combination of the basis functions on a MeshCell.
        """
        abs_cell_idx = self.V.T.get_abs_cell_idx(cell_idx)
        int_vals = self.interior_values[abs_cell_idx]
        vals = zeros(shape(int_vals[0]))
        for i, val in enumerate(int_vals):
            vals += val * coef[i]
        return vals

    def get_coef_on_mesh(self, cell_idx: int, u: ndarray) -> ndarray:
        """
        Get the coefficients of the basis functions on a MeshCell.
        """
        abs_cell_idx = self.V.T.get_abs_cell_idx(cell_idx)
        keys = self.V.cell_dofs[abs_cell_idx]
        coef = zeros((len(keys),))
        for i, key in enumerate(keys):
            coef[i] = u[key.glob_idx]
        return coef
