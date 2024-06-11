"""
Solve the global linear system.

Classes
-------
Solver
    Solve the global linear system.
"""

from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from ..locfun.local_space import LocalPoissonFunction
from ..util.print_color import Color, print_color
from .bilinear_form import BilinearForm
from .globfunsp import GlobalFunctionSpace
from .globkey import GlobalKey


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
    # interior_values : list[list[np.ndarray]]
    #     List of interior values on each MeshCell
    soln : np.ndarray
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
    local_function_spaces: list[Optional[LocalPoissonFunction]]
    soln: np.ndarray

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

    def get_solution(self) -> np.ndarray:
        """
        Get the solution vector.

        Returns
        -------
        np.ndarray
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

    def _build_values_and_indexes(
        self,
        verbose: bool,
        compute_interior_values: bool,
        compute_interior_gradient: bool,
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
        self._initialize_linear_system_arrays()

        # loop over MeshCells
        for abs_cell_idx in range(self.glob_fun_sp.mesh.num_cells):

            if verbose:
                self._print_cell_progress(abs_cell_idx)

            # build local function space
            loc_fun_sp = self.glob_fun_sp.build_local_function_space(
                cell_idx=self.glob_fun_sp.mesh.cell_idx_list[abs_cell_idx],
                verbose=verbose,
                compute_interior_values=compute_interior_values,
                compute_interior_gradient=compute_interior_gradient,
            )

            if compute_interior_values:
                self.local_function_spaces[abs_cell_idx] = loc_fun_sp

            if verbose:
                print("Evaluating bilinear form and right-hand side...")

            # get local basis functions
            loc_basis = loc_fun_sp.get_basis()

            # double loop over local basis functions
            for i in self._get_range_num_funs(verbose, loc_fun_sp.num_funs):
                v = loc_basis[i]
                self._set_rhs_values(v)
                for j in range(i, loc_fun_sp.num_funs):
                    w = loc_basis[j]
                    self._compute_and_set_matrix_values(v, w, j > i)

        # impose zero Dirichlet boundary conditions
        self._impose_zero_dirichlet_bc()

    def _initialize_linear_system_arrays(self) -> None:
        self.rhs_idx = []
        self.rhs_vals = []
        self.row_idx = []
        self.col_idx = []
        self.mat_vals = []
        self.stiff_vals = []
        self.mass_vals = []
        self.local_function_spaces = [
            None for _ in range(self.glob_fun_sp.mesh.num_cells)
        ]

    def _print_cell_progress(self, abs_cell_idx: int) -> None:
        print_color(
            "Cell "
            + f"{abs_cell_idx + 1:6}"
            + f" / {self.glob_fun_sp.mesh.num_cells:6}",
            Color.GREEN,
        )

    def _get_range_num_funs(
        self, verbose: bool, num_funs: int
    ) -> Union[range, tqdm]:
        if verbose:
            return tqdm(range(num_funs))
        return range(num_funs)

    def _set_rhs_values(self, v: LocalPoissonFunction) -> None:
        f_i = self.bilinear_form.eval_rhs(v)
        self.rhs_idx.append(v.key.glob_idx)
        self.rhs_vals.append(f_i)

    def _compute_and_set_matrix_values(
        self,
        v: LocalPoissonFunction,
        w: LocalPoissonFunction,
        apply_symmetry: bool,
    ) -> None:
        # TODO: storing stiffness and mass matrices separately is inefficient

        # evaluate local bilinear form
        h1_ij = self.bilinear_form.eval_h1(v, w)
        l2_ij = self.bilinear_form.eval_l2(v, w)
        a_ij = self.bilinear_form.eval_with_h1_and_l2(h1_ij, l2_ij)

        # add to matrices
        i = v.key.glob_idx
        j = w.key.glob_idx
        self._set_matrix_values(i, j, a_ij, h1_ij, l2_ij)
        if apply_symmetry:
            self._set_matrix_values(
                i=j, j=i, a_ij=a_ij, h1_ij=h1_ij, l2_ij=l2_ij
            )

    def _set_matrix_values(
        self, i: int, j: int, a_ij: float, h1_ij: float, l2_ij: float
    ) -> None:
        self.row_idx.append(i)
        self.col_idx.append(j)
        self.mat_vals.append(a_ij)
        self.stiff_vals.append(h1_ij)
        self.mass_vals.append(l2_ij)

    def _impose_zero_dirichlet_bc(self) -> None:
        for abs_cell_idx in range(self.glob_fun_sp.mesh.num_cells):
            for key in self.glob_fun_sp.cell_dofs[abs_cell_idx]:
                if key.is_on_boundary:
                    self._set_rhs_from_key(key, 0.0)
                    self._zero_mat_row(key)

    def _set_rhs_from_key(self, key: GlobalKey, val: float) -> None:
        for k, idx in enumerate(self.rhs_idx):
            if idx == key.glob_idx:
                self.rhs_vals[k] = val

    def _zero_mat_row(self, key: GlobalKey) -> None:
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
        self._check_indices_do_not_exceed_num_funs(self.rhs_idx, "rhs_idx")
        self._check_indices_do_not_exceed_num_funs(self.row_idx, "row_idx")
        self._check_indices_do_not_exceed_num_funs(self.col_idx, "col_idx")

    def _check_indices_do_not_exceed_num_funs(
        self, indices: list[int], list_name: str
    ) -> None:
        size = max(indices)
        if size >= self.num_funs:
            raise ValueError(
                f"{list_name} >= self.num_funs ({size} >= {self.num_funs})"
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

    def compute_linear_combo_on_mesh_cell(
        self, cell_idx: int, global_coef: np.ndarray
    ) -> LocalPoissonFunction:
        """
        Compute a linear combination of the basis functions on a MeshCell.

        Parameters
        ----------
        cell_idx : int
            MeshCell index
        coef : np.ndarray
            Coefficients
        """
        abs_cell_idx = self.glob_fun_sp.mesh.get_abs_cell_idx(cell_idx)
        loc_fun_sp = self.local_function_spaces[abs_cell_idx]
        local_coef = self.get_coef_on_mesh_cell(cell_idx, global_coef)
        w = LocalPoissonFunction(nyst=loc_fun_sp.nyst, evaluate_gradient=True)
        for i, v in enumerate(loc_fun_sp.get_basis()):
            w += v * local_coef[i]
        return w

    def get_coef_on_mesh_cell(self, cell_idx: int, global_coef: np.ndarray) -> np.ndarray:
        """
        Get the coefficients of the basis functions on a MeshCell.

        Parameters
        ----------
        cell_idx : int
            MeshCell index
        global_coef : np.ndarray
            Solution vector
        """
        abs_cell_idx = self.glob_fun_sp.mesh.get_abs_cell_idx(cell_idx)
        keys = self.glob_fun_sp.cell_dofs[abs_cell_idx]
        local_coef = np.zeros((len(keys),))
        for i, key in enumerate(keys):
            local_coef[i] = global_coef[key.glob_idx]
        return local_coef
