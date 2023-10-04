"""
solver.py
=========

Module containing the solver class, which is a convenience class for solving
the global linear system.
"""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from numpy import inf, nanmax, nanmin, ndarray, shape, zeros
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from ..util.print_color import Color, print_color
from .bilinear_form import bilinear_form
from .globfunsp import global_function_space


class solver:
    """
    Convenience class for solving the global linear system.

    Attributes
    ----------
    V : global_function_space
        Global function space
    a : bilinear_form
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
        List of interior values on each cell
    soln : ndarray
        Solution vector
    """

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
    soln: ndarray

    def __init__(self, V: global_function_space, a: bilinear_form) -> None:
        """
        Constructor for solver class.

        Parameters
        ----------
        V : global_function_space
            Global function space
        a : bilinear_form
            Bilinear form
        """
        self.set_global_function_space(V)
        self.set_bilinear_form(a)

    def __str__(self) -> str:
        """
        Return string representation.
        """
        s = "solver:\n"
        s += f"\tV: {self.V}\n"
        s += f"\ta: {self.a}\n"
        s += f"\tnum_funs: {self.num_funs}\n"
        return s

    # SETTERS ################################################################

    def set_global_function_space(self, V: global_function_space) -> None:
        """
        Set the global function space.
        """
        if not isinstance(V, global_function_space):
            raise TypeError("V must be a global_function_space")
        self.V = V

    def set_bilinear_form(self, a: bilinear_form) -> None:
        """
        Set the bilinear form.
        """
        if not isinstance(a, bilinear_form):
            raise TypeError("a must be a bilinear_form")
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

    def assemble(self, verbose: bool = True, processes: int = 1) -> None:
        """Assemble global system matrix"""
        self.build_values_and_indexes(verbose=verbose, processes=processes)
        self.find_num_funs()
        self.build_matrix_and_rhs()

    def build_values_and_indexes(
        self, verbose: bool = True, processes: int = 1
    ) -> None:
        """Build values and indexes"""
        if processes == 1:
            self.build_values_and_indexes_sequential(verbose=verbose)
        elif processes > 1:
            self.build_values_and_indexes_parallel(
                verbose=verbose, processes=processes
            )
        else:
            raise ValueError("processes must be a positive integer")

    def build_values_and_indexes_parallel(
        self, verbose: bool = True, processes: int = 1
    ) -> None:
        """
        Build values and indexes in parallel.
        """
        raise NotImplementedError("Parallel assembly not yet implemented")

    def build_values_and_indexes_sequential(self, verbose: bool = True) -> None:
        """
        Build values and indexes sequentially.
        """

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
                    f"Cell {abs_cell_idx + 1:6} / {self.V.T.num_cells:6}",
                    Color.GREEN,
                )

            # build local function space
            V_K = self.V.build_local_function_space(cell_idx, verbose=verbose)

            # initialize interior values
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
                self.interior_values[abs_cell_idx][i] = v.int_vals

                # evaluate local right-hand side
                f_i = self.a.eval_rhs(v)

                # add to global right-hand side vector
                self.rhs_idx.append(v.key.glob_idx)
                self.rhs_vals.append(f_i)

                for j in range(i, V_K.num_funs):
                    w = loc_basis[j]

                    # evaluate local bilinear form
                    a_ij = self.a.eval(v, w)

                    # add to global stiffness matrix
                    self.row_idx.append(v.key.glob_idx)
                    self.col_idx.append(w.key.glob_idx)
                    self.mat_vals.append(a_ij)

                    # symmetry
                    if j > i:
                        self.row_idx.append(w.key.glob_idx)
                        self.col_idx.append(v.key.glob_idx)
                        self.mat_vals.append(a_ij)

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

    # PLOT SOLUTION ##########################################################

    # TODO: this belongs in a different module

    def plot_solution(
        self,
        title: str = "",
        show_fig: bool = True,
        save_fig: bool = False,
        filename: str = "solution.pdf",
        fill: bool = True,
    ) -> None:
        """
        Plot the solution.
        """
        self.plot_linear_combo(
            self.soln,
            title=title,
            show_fig=show_fig,
            save_fig=save_fig,
            filename=filename,
            fill=fill,
        )

    def plot_linear_combo(
        self,
        u: ndarray,
        title: str = "",
        show_fig: bool = True,
        save_fig: bool = False,
        filename: str = "solution.pdf",
        fill: bool = True,
    ) -> None:
        """
        Plot a linear combination of the basis functions.
        """
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
        min_x, max_x, min_y, max_y = self.get_axis_limits()
        w, h = self.get_figure_size(min_x, max_x, min_y, max_y)

        # get figure object
        fig = plt.figure(figsize=(w, h))

        # plot mesh edges
        for e in self.V.T.edges:
            plt.plot(e.x[0, :], e.x[1, :], "k")

        # plot interior values on each cell
        for cell_idx in self.V.T.cell_idx_list:
            vals = vals_arr[cell_idx]
            if vmax - vmin > 1e-6:
                K = self.V.T.get_cell(cell_idx)
                abs_cell_idx = self.V.T.get_abs_cell_idx(cell_idx)
                K.parameterize(self.V.quad_dict)
                if fill:
                    plt.contourf(
                        K.int_x1,
                        K.int_x2,
                        vals_arr[abs_cell_idx],
                        vmin=vmin,
                        vmax=vmax,
                        levels=32,
                    )
                else:
                    plt.contour(
                        K.int_x1,
                        K.int_x2,
                        vals_arr[abs_cell_idx],
                        vmin=vmin,
                        vmax=vmax,
                        levels=32,
                        colors="b",
                    )
        if fill:
            sm = ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax))
            plt.colorbar(
                mappable=sm,
                ax=fig.axes,
                fraction=0.046,
                pad=0.04,
            )
        plt.axis("equal")
        plt.axis("off")
        plt.gca().set_aspect("equal")
        plt.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0
        )

        if len(title) > 0:
            plt.title(title)

        if save_fig:
            plt.savefig(filename)

        if show_fig:
            plt.show()

        plt.close(fig)

    def get_axis_limits(self) -> tuple[float, float, float, float]:
        """
        Get the axis limits.
        """
        min_x = inf
        max_x = -inf
        min_y = inf
        max_y = -inf
        for e in self.V.T.edges:
            min_x = self.update_min(min_x, e.x[0, :])
            max_x = self.update_max(max_x, e.x[0, :])
            min_y = self.update_min(min_y, e.x[1, :])
            max_y = self.update_max(max_y, e.x[1, :])
        return min_x, max_x, min_y, max_y

    def update_min(self, current_min: float, candidates: ndarray) -> float:
        """
        Update the minimum value.
        """
        min_candidate = min(candidates)
        new_min = min(current_min, min_candidate)
        return new_min

    def update_max(self, current_max: float, candidates: ndarray) -> float:
        """
        Update the maximum value.
        """
        max_candidate = max(candidates)
        new_max = max(current_max, max_candidate)
        return new_max

    def get_figure_size(
        self, min_x: float, max_x: float, min_y: float, max_y: float
    ) -> tuple[float, float]:
        """
        Get the figure size.
        """
        dx = max_x - min_x
        dy = max_y - min_y
        h = 4.0
        w = h * dx / dy
        return w, h

    def compute_linear_combo_on_cell(
        self, cell_idx: int, coef: ndarray
    ) -> ndarray:
        """
        Compute a linear combination of the basis functions on a cell.
        """
        abs_cell_idx = self.V.T.get_abs_cell_idx(cell_idx)
        int_vals = self.interior_values[abs_cell_idx]
        vals = zeros(shape(int_vals[0]))
        for i, val in enumerate(int_vals):
            vals += val * coef[i]
        return vals

    def get_coef_on_cell(self, cell_idx: int, u: ndarray) -> ndarray:
        """
        Get the coefficients of the basis functions on a cell.
        """
        abs_cell_idx = self.V.T.get_abs_cell_idx(cell_idx)
        keys = self.V.cell_dofs[abs_cell_idx]
        coef = zeros((len(keys),))
        for i, key in enumerate(keys):
            coef[i] = u[key.glob_idx]
        return coef
