"""
Solve the global linear system.

Classes
-------
Solver
    Solve the global linear system.
"""

from typing import Union

from numpy import ndarray, shape, zeros
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from ..util.print_color import Color, print_color
from .bilinear_form import BilinearForm
from .globfunsp import GlobalFunctionSpace


class Solver:
    """
    Solve the global linear system.

    Attributes
    ----------
    glob_fun_sp : GlobalFunctionSpace
        Global function space
    bilinear_form : BilinearForm
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

    glob_fun_sp: GlobalFunctionSpace
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

    def __init__(
        self,
        glob_fun_sp: GlobalFunctionSpace,
        bilinear_form: BilinearForm,
        verbose: bool = True,
        compute_interior_values: bool = True,
    ) -> None:
        """
        Initialize the Solver.

        Parameters
        ----------
        glob_fun_sp : GlobalFunctionSpace
            Global function space
        bilinear_form : BilinearForm
            Bilinear form
        verbose : bool, optional
            Print progress, by default True
        compute_interior_values : bool, optional
            Compute interior values, by default True
        """
        self.set_global_function_space(glob_fun_sp)
        self.set_bilinear_form(bilinear_form)
        self.assemble(
            verbose=verbose, compute_interior_values=compute_interior_values
        )

    def __str__(self) -> str:
        """Get string representation."""
        s = "Solver:\n"
        s += f"\tglob_fun_sp: {self.glob_fun_sp}\n"
        s += f"\ta: {self.bilinear_form}\n"
        s += f"\tnum_funs: {self.num_funs}\n"
        return s

    # SETTERS ################################################################

    def set_global_function_space(
        self, glob_fun_sp: GlobalFunctionSpace
    ) -> None:
        """
        Set the global function space.

        Parameters
        ----------
        glob_fun_sp : GlobalFunctionSpace
            Global function space
        """
        if not isinstance(glob_fun_sp, GlobalFunctionSpace):
            raise TypeError("glob_fun_sp must be a GlobalFunctionSpace")
        self.glob_fun_sp = glob_fun_sp

    def set_bilinear_form(self, bilinear_form: BilinearForm) -> None:
        """
        Set the bilinear form.

        Parameters
        ----------
        a : BilinearForm
            Bilinear form
        """
        if not isinstance(bilinear_form, BilinearForm):
            raise TypeError("bilinear_form must be a BilinearForm")
        self.bilinear_form = bilinear_form

    # SOLVE LINEAR SYSTEM ####################################################

    def solve(self) -> None:
        """Solve the linear system."""
        self.soln = spsolve(self.glob_mat, self.glob_rhs)

    def get_solution(self) -> ndarray:
        """
        Get the solution vector.

        Returns
        -------
        ndarray
            Solution vector
        """
        if self.soln is None:
            self.solve()
        return self.soln

    # ASSEMBLE GLOBAL SYSTEM #################################################

    def assemble(
        self,
        verbose: bool = True,
        compute_interior_values: bool = True,
        compute_interior_gradient: bool = False,
    ) -> None:
        """
        Assemble the global system matrix and right-hand side vector.

        Parameters
        ----------
        verbose : bool, optional
            Print progress, by default True
        processes : int, optional
            Number of processes to use, by default 1
        compute_interior_values : bool, optional
            Compute interior values, by default True
        """
        self._build_values_and_indexes(
            verbose=verbose,
            compute_interior_values=compute_interior_values,
            compute_interior_gradient=compute_interior_gradient,
        )
        self._find_num_funs()
        self._build_matrix_and_rhs()
        self.build_stiffness_matrix()
        self.build_mass_matrix()

    def build_values_and_indexes(
        self,
        verbose: bool = True,
        compute_interior_values: bool = True,
        compute_interior_gradient: bool = False,
    ) -> None:
        """
        Build values and indexes.

        Parameters
        ----------
        verbose : bool, optional
            Print progress, by default True
        compute_interior_values : bool, optional
            Compute interior values, by default True
        """
        self.rhs_idx = []
        self.rhs_vals = []
        self.row_idx = []
        self.col_idx = []
        self.mat_vals = []
        self.stiff_vals = []
        self.mass_vals = []

        self.interior_values = [
            [] for _ in range(self.glob_fun_sp.mesh.num_cells)
        ]

        # loop over MeshCells
        for abs_cell_idx in range(self.glob_fun_sp.mesh.num_cells):
            cell_idx = self.glob_fun_sp.mesh.cell_idx_list[abs_cell_idx]

            if verbose:
                print_color(
                    "Cell "
                    + f"{abs_cell_idx + 1:6}"
                    + f" / {self.glob_fun_sp.mesh.num_cells:6}",
                    Color.GREEN,
                )

            # build local function space
            loc_fun_sp = self.glob_fun_sp.build_local_function_space(
                cell_idx,
                verbose=verbose,
                compute_interior_values=compute_interior_values,
                compute_interior_gradient=compute_interior_gradient,
            )

            # initialize interior values
            if compute_interior_values:
                self.interior_values[abs_cell_idx] = [
                    zeros((0,)) for _ in range(loc_fun_sp.num_funs)
                ]

            if verbose:
                print("Evaluating bilinear form and right-hand side...")

            # loop over local functions
            loc_basis = loc_fun_sp.get_basis()

            range_num_funs: Union[range, tqdm]
            if verbose:
                range_num_funs = tqdm(range(loc_fun_sp.num_funs))
            else:
                range_num_funs = range(loc_fun_sp.num_funs)

            for i in range_num_funs:
                v = loc_basis[i]

                # store interior values
                if compute_interior_values:
                    self.interior_values[abs_cell_idx][i] = v.int_vals

                # evaluate local right-hand side
                f_i = self.bilinear_form.eval_rhs(v)

                # add to global right-hand side vector
                self.rhs_idx.append(v.key.glob_idx)
                self.rhs_vals.append(f_i)

                for j in range(i, loc_fun_sp.num_funs):
                    w = loc_basis[j]

                    # evaluate local bilinear form
                    h1_ij = self.bilinear_form.eval_h1(v, w)
                    l2_ij = self.bilinear_form.eval_l2(v, w)
                    a_ij = self.bilinear_form.eval_with_h1_and_l2(h1_ij, l2_ij)

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

        # impose boundary conditions
        for abs_cell_idx in range(self.glob_fun_sp.mesh.num_cells):
            cell_idx = self.glob_fun_sp.mesh.cell_idx_list[abs_cell_idx]

            for key in self.glob_fun_sp.cell_dofs[abs_cell_idx]:
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

    def _find_num_funs(self) -> None:
        self.num_funs = self.glob_fun_sp.num_funs
        self._check_sizes()

    def _check_sizes(self) -> None:
        rhs_size = 1 + max(self.rhs_idx)
        if rhs_size > self.num_funs:
            raise ValueError(
                f"rhs_size > self.num_funs ({rhs_size} > {self.num_funs})"
            )

        num_rows = 1 + max(self.row_idx)
        if num_rows > self.num_funs:
            raise ValueError(
                f"num_rows > self.num_funs ({num_rows} > {self.num_funs})"
            )

        num_cols = 1 + max(self.col_idx)
        if num_cols > self.num_funs:
            raise ValueError(
                f"num_cols > self.num_funs ({num_cols} > {self.num_funs})"
            )

    def _build_matrix_and_rhs(self) -> None:
        self.glob_mat = csr_matrix(
            (self.mat_vals, (self.row_idx, self.col_idx)),
            shape=(self.num_funs, self.num_funs),
        )
        self.glob_rhs = csr_matrix(
            (self.rhs_vals, (self.rhs_idx, [0] * len(self.rhs_idx))),
            shape=(self.num_funs, 1),
        )

    def build_stiffness_matrix(self) -> None:
        """
        Build stiffness matrix, A_ij = int_Omega grad v_i * grad v_j dx.

        Result is stored in self.stiff_mat.
        """
        self.stiff_mat = csr_matrix(
            (self.stiff_vals, (self.row_idx, self.col_idx)),
            shape=(self.num_funs, self.num_funs),
        )

    def build_mass_matrix(self) -> None:
        """
        Build mass matrix, M_ij = int_Omega v_i * v_j dx.

        Result is stored in self.mass_mat.
        """
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

        Parameters
        ----------
        cell_idx : int
            MeshCell index
        coef : ndarray
            Coefficients
        """
        abs_cell_idx = self.glob_fun_sp.mesh.get_abs_cell_idx(cell_idx)
        int_vals = self.interior_values[abs_cell_idx]
        vals = zeros(shape(int_vals[0]))
        for i, val in enumerate(int_vals):
            vals += val * coef[i]
        return vals

    def get_coef_on_mesh(self, cell_idx: int, u: ndarray) -> ndarray:
        """
        Get the coefficients of the basis functions on a MeshCell.

        Parameters
        ----------
        cell_idx : int
            MeshCell index
        u : ndarray
            Solution vector
        """
        abs_cell_idx = self.glob_fun_sp.mesh.get_abs_cell_idx(cell_idx)
        keys = self.glob_fun_sp.cell_dofs[abs_cell_idx]
        coef = zeros((len(keys),))
        for i, key in enumerate(keys):
            coef[i] = u[key.glob_idx]
        return coef
