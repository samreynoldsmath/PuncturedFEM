"""
Plot global functions.

Classes
-------
GlobalFunctionPlot
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap
from numpy import inf, nanmax, nanmin, ndarray

from ..solver.solver import Solver
from .plot_util import get_axis_limits, get_figure_size, save_figure

# from .locfun_plot import LocalFunctionPlot


class GlobalFunctionPlot:
    """
    Plot a global function.

    Attributes
    ----------
    solver : Solver
        A Solver object, which contains the Mesh and the basis functions.
    coef : ndarray
        The coefficients of the linear combination of basis functions.
    fill : bool
        If True, a heatmap is plotted. If False, a contour plot is plotted.
    title : str
        The title of the plot.
    show_colorbar : bool
        If True, a colorbar is shown.
    show_axis : bool
        If True, the axis is shown.
    colormap : Optional[Colormap]
        The colormap used for the plot.
    """

    solver: Solver
    coef: ndarray
    fill: bool
    title: str
    show_colorbar: bool
    show_axis: bool
    colormap: Optional[Colormap]

    def __init__(self, solver: Solver, coef: Optional[ndarray] = None) -> None:
        """
        Initialize a GlobalFunctionPlot object.

        Parameters
        ----------
        solver : Solver
            A Solver object, which contains the Mesh and the basis functions.
        coef : ndarray
            The coefficients of the linear combination of basis functions.
        """
        self.set_solver(solver)
        self.set_coefficients(coef)

    def set_solver(self, solver: Solver) -> None:
        """
        Set the solver object.

        Parameters
        ----------
        solver : Solver
            A Solver object, which contains the Mesh and the basis functions.
        """
        if not isinstance(solver, Solver):
            raise TypeError("solver must be of type Solver")
        self.solver = solver

    def set_coefficients(self, coef: Optional[ndarray] = None) -> None:
        """
        Set the coefficients of the linear combination of basis functions.

        Parameters
        ----------
        coef : ndarray, optional
            The coefficients of the linear combination of basis functions.
            Default is None. If None, the solver object must have an attribute
            soln.
        """
        if coef is None:
            if not hasattr(self.solver, "soln"):
                raise AttributeError(
                    "solver must have attribute soln or u must be given"
                )
            coef = self.solver.soln
        if not isinstance(coef, ndarray):
            raise TypeError("u must be of type ndarray")
        if coef.shape != (self.solver.glob_fun_sp.num_funs,):
            raise ValueError("u must have shape (solver.glob_fun_sp.num_funs,)")
        self.coef = coef

    def _unpack_kwargs(self, kwargs: dict) -> None:
        self.fill = kwargs.get("fill", True)
        self.title = kwargs.get("title", "")
        self.show_colorbar = kwargs.get("show_colorbar", True)
        self.show_axis = kwargs.get("show_axis", True)
        self.colormap = kwargs.get("colormap", None)

    def draw(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        """
        Draw the plot.

        Parameters
        ----------
        show_plot : bool, optional
            If True, the plot is shown. Default is True.
        filename : str, optional
            If not empty, the plot is saved to this file. Default is "".

        Other Parameters
        ----------------
        fill : bool, optional
            If True, a heatmap is plotted. If False, a contour plot is plotted.
            Default is True.
        title : str, optional
            The title of the plot. Default is "", i.e. no title.
        show_colorbar : bool, optional
            If True, a colorbar is shown. Default is True.
        show_axis : bool, optional
            If True, the axis is shown. Default is True.
        colormap : Optional[Colormap], optional
            The colormap used for the plot. Default is None.
        """
        self._unpack_kwargs(kwargs)
        _plot_linear_combo(
            self.solver,
            self.coef,
            title=self.title,
            show_fig=show_plot,
            save_fig=(len(filename) > 0),
            filename=filename,
            fill=self.fill,
            show_colorbar=self.show_colorbar,
            colormap=self.colormap,
        )


def _plot_linear_combo(
    solver: Solver,
    u: ndarray,
    title: str = "",
    show_fig: bool = True,
    save_fig: bool = False,
    filename: str = "solution.pdf",
    fill: bool = True,
    show_colorbar: bool = True,
    colormap: Optional[Colormap] = None,
) -> None:
    if not (show_fig or save_fig):
        return
    # compute linear combo on each MeshCell, determine range of global values
    vals_arr = []
    v_min = inf
    v_max = -inf
    for cell_idx in solver.glob_fun_sp.mesh.cell_idx_list:
        coef = solver.get_coef_on_mesh(cell_idx, u)
        vals = solver.compute_linear_combo_on_mesh(cell_idx, coef)
        vals_arr.append(vals)
        v_min = min(v_min, nanmin(vals))
        v_max = max(v_max, nanmax(vals))

    # determine axes and figure size
    min_x, max_x, min_y, max_y = get_axis_limits(solver.glob_fun_sp.mesh.edges)
    w, h = get_figure_size(min_x, max_x, min_y, max_y)

    # get figure object
    fig = plt.figure(figsize=(w, h))

    # plot mesh edges
    for e in solver.glob_fun_sp.mesh.edges:
        plt.plot(e.x[0, :], e.x[1, :], "k")

    vals = np.concatenate(vals_arr)
    x1 = np.concatenate(solver.interior_x1)
    x2 = np.concatenate(solver.interior_x2)

    triang = tri.Triangulation(x1, x2)
    plt.tricontourf(
        triang,
        vals,
        vmin=v_min,
        vmax=v_max,
        levels=32,
        cmap=colormap,
    )
    if fill and show_colorbar:
        sm = ScalarMappable(
            norm=plt.Normalize(vmin=v_min, vmax=v_max), cmap=colormap
        )
        plt.colorbar(
            mappable=sm,
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
        save_figure(filename)

    if show_fig:
        plt.show()

    plt.close(fig)
