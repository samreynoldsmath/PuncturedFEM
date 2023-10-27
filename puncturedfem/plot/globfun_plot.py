"""
plot_global_solution.py
=======================

Module for plotting the global solution.
"""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from numpy import inf, nanmax, nanmin, ndarray

from ..solver.solver import Solver


class GlobalFunctionPlot:
    """
    Class for plotting a global function.

    Parameters
    ----------
    solver : Solver
        A Solver object, which contains the Mesh and the basis functions.
    u : ndarray
        The coefficients of the linear combination of basis functions.
    fill : bool, optional
        If True, a heatmap is plotted. If False, a contour plot is plotted.
        Default is True.
    title : str, optional
        The title of the plot. Default is "", i.e. no title.
    """

    solver: Solver
    u: ndarray
    fill: bool
    title: str

    def __init__(
        self,
        solver: Solver,
        u: ndarray,
        fill: bool = True,
        title: str = "",
    ) -> None:
        """
        Constructor for the GlobalFunctionPlot class.

        Parameters
        ----------
        solver : Solver
            A Solver object, which contains the Mesh and the basis functions.
        u : ndarray
            The coefficients of the linear combination of basis functions.
        fill : bool, optional
            If True, a heatmap is plotted. If False, a contour plot is plotted.
            Default is True.
        title : str, optional
            The title of the plot. Default is "", i.e. no title.
        """
        self.set_solver(solver)
        self.set_coefficients(u)
        self.fill = fill
        self.title = title

    def set_solver(self, solver: Solver) -> None:
        """
        Set the solver.
        """
        if not isinstance(solver, Solver):
            raise TypeError("solver must be of type Solver")
        self.solver = solver

    def set_coefficients(self, u: ndarray) -> None:
        """
        Set the coefficients.
        """
        if not isinstance(u, ndarray):
            raise TypeError("u must be of type ndarray")
        if u.shape != (self.solver.V.num_funs,):
            raise ValueError("u must have shape (solver.V.N,)")
        self.u = u

    def draw(self, show_plot: bool = True, filename: str = "") -> None:
        """
        Draw the plot.

        Parameters
        ----------
        show_plot : bool, optional
            If True, the plot is shown. Default is True.
        filename : str, optional
            If not empty, the plot is saved to this file. Default is "".
        """
        _plot_linear_combo(
            self.solver,
            self.u,
            title=self.title,
            show_fig=show_plot,
            save_fig=(len(filename) > 0),
            filename=filename,
            fill=self.fill,
        )


def _plot_linear_combo(
    solver: Solver,
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
    # compute linear combo on each MeshCell, determine range of global values
    vals_arr = []
    v_min = inf
    v_max = -inf
    for cell_idx in solver.V.T.cell_idx_list:
        coef = solver.get_coef_on_mesh(cell_idx, u)
        vals = solver.compute_linear_combo_on_mesh(cell_idx, coef)
        vals_arr.append(vals)
        v_min = min(v_min, nanmin(vals))
        v_max = max(v_max, nanmax(vals))

    # determine axes and figure size
    min_x, max_x, min_y, max_y = _get_axis_limits(solver)
    w, h = _get_figure_size(min_x, max_x, min_y, max_y)

    # get figure object
    fig = plt.figure(figsize=(w, h))

    # plot mesh edges
    for e in solver.V.T.edges:
        plt.plot(e.x[0, :], e.x[1, :], "k")

    # plot interior values on each MeshCell
    for cell_idx in solver.V.T.cell_idx_list:
        vals = vals_arr[cell_idx]
        if v_max - v_min > 1e-6:
            K = solver.V.T.get_cells(cell_idx)
            abs_cell_idx = solver.V.T.get_abs_cell_idx(cell_idx)
            K.parameterize(solver.V.quad_dict)
            if fill:
                plt.contourf(
                    K.int_x1,
                    K.int_x2,
                    vals_arr[abs_cell_idx],
                    v_min=v_min,
                    v_max=v_max,
                    levels=32,
                )
            else:
                plt.contour(
                    K.int_x1,
                    K.int_x2,
                    vals_arr[abs_cell_idx],
                    v_min=v_min,
                    v_max=v_max,
                    levels=32,
                    colors="b",
                )
    if fill:
        sm = ScalarMappable(norm=plt.Normalize(vmin=v_min, vmax=v_max))
        plt.colorbar(
            mappable=sm,
            # ax=fig.axes,
            ax=plt.gca(),
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


def _get_axis_limits(solver: Solver) -> tuple[float, float, float, float]:
    """
    Get the axis limits.
    """
    min_x = inf
    max_x = -inf
    min_y = inf
    max_y = -inf
    for e in solver.V.T.edges:
        min_x = _update_min(min_x, e.x[0, :])
        max_x = _update_max(max_x, e.x[0, :])
        min_y = _update_min(min_y, e.x[1, :])
        max_y = _update_max(max_y, e.x[1, :])
    return min_x, max_x, min_y, max_y


def _update_min(current_min: float, candidates: ndarray) -> float:
    """
    Update the minimum value.
    """
    min_candidate = min(candidates)
    return min(current_min, min_candidate)


def _update_max(current_max: float, candidates: ndarray) -> float:
    """
    Update the maximum value.
    """
    max_candidate = max(candidates)
    return max(current_max, max_candidate)


def _get_figure_size(
    min_x: float, max_x: float, min_y: float, max_y: float
) -> tuple[float, float]:
    """
    Get the figure size.
    """
    dx = max_x - min_x
    dy = max_y - min_y
    h = 4.0
    w = h * dx / dy
    return w, h
