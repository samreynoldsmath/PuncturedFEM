"""
Plot global functions.

Classes
-------
GlobalFunctionPlot
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

from ..locfun.local_poisson import LocalPoissonFunction
from ..solver.solver import Solver
from .locfun_plot import LocalFunctionPlot
from .plot_util import get_axis_limits, get_figure_size, save_figure

# from matplotlib.colors import Colormap


class GlobalFunctionPlot:
    """
    Plot a global function.

    Attributes
    ----------
    solver : Solver
        A Solver object, which contains the Mesh and the basis functions.
    coef : np.ndarray
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
    coef: np.ndarray
    global_function: list[LocalPoissonFunction]
    global_range: tuple[float, float]
    global_grad1_range: tuple[float, float]
    global_grad2_range: tuple[float, float]
    global_grad_norm_range: tuple[float, float]

    def __init__(
        self, solver: Solver, coef: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize a GlobalFunctionPlot object.

        Parameters
        ----------
        solver : Solver
            A Solver object, which contains the Mesh and the basis functions.
        coef : np.ndarray
            The coefficients of the linear combination of basis functions.
        """
        self.set_solver(solver)
        self.set_coefficients(coef)
        self._build_global_function()

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

    def set_coefficients(self, coef: Optional[np.ndarray] = None) -> None:
        """
        Set the coefficients of the linear combination of basis functions.

        Parameters
        ----------
        coef : np.ndarray, optional
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
        if not isinstance(coef, np.ndarray):
            raise TypeError("u must be of type np.ndarray")
        if coef.shape != (self.solver.glob_fun_sp.num_funs,):
            raise ValueError("u must have shape (solver.glob_fun_sp.num_funs,)")
        self.coef = coef

    def _build_global_function(self) -> None:
        self.global_function = []
        self.global_range = (np.inf, -np.inf)
        self.global_grad1_range = (np.inf, -np.inf)
        self.global_grad2_range = (np.inf, -np.inf)
        self.global_grad_norm_range = (np.inf, -np.inf)
        for cell_idx in self.solver.glob_fun_sp.mesh.cell_idx_list:
            local_fun = self.solver.compute_linear_combo_on_mesh_cell(
                cell_idx, self.coef
            )
            self.global_function.append(local_fun)
            self.global_range = _range_on_cell(
                local_fun.int_vals, self.global_range[0], self.global_range[1]
            )
            self.global_grad1_range = _range_on_cell(
                local_fun.int_grad1,
                self.global_grad1_range[0],
                self.global_grad1_range[1],
            )
            self.global_grad2_range = _range_on_cell(
                local_fun.int_grad2,
                self.global_grad2_range[0],
                self.global_grad2_range[1],
            )
            grad_norm = np.sqrt(local_fun.int_grad1**2 + local_fun.int_grad2**2)
            self.global_grad_norm_range = _range_on_cell(
                grad_norm,
                self.global_grad_norm_range[0],
                self.global_grad_norm_range[1],
            )

    def draw(
        self,
        plot_type: str = "values",
        show_plot: bool = True,
        filename: str = "",
        **kwargs: Any,
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
        fill = kwargs.get("fill", True)
        title = kwargs.get("title", "")
        show_colorbar = kwargs.get("show_colorbar", False)
        show_axis = kwargs.get("show_axis", False)
        colormap = kwargs.get("colormap", None)
        use_interp = kwargs.get("use_interp", True)

        save_fig = len(filename) > 0
        if not (show_plot or save_fig):
            return

        # determine axes and figure size
        min_x, max_x, min_y, max_y = get_axis_limits(
            self.solver.glob_fun_sp.mesh.edges
        )
        w, h = get_figure_size(min_x, max_x, min_y, max_y)

        # get figure object
        fig = plt.figure(figsize=(w, h))

        # determine global range
        if plot_type == "values":
            global_min, global_max = self.global_range
        elif plot_type == "grad_x1":
            global_min, global_max = self.global_grad1_range
        elif plot_type == "grad_x2":
            global_min, global_max = self.global_grad2_range
        elif plot_type == "grad_norm":
            global_min, global_max = self.global_grad_norm_range
        else:
            raise ValueError(
                f"plot_type must be 'values', 'grad_x1', 'grad_x2', or 'grad_norm', not '{plot_type}'"
            )

        # plot linear combination on each MeshCell
        for local_fun in self.global_function:

            # plot local function
            plot = LocalFunctionPlot(local_fun, use_interp)
            plot.draw(
                plot_type=plot_type,
                fill=fill,
                new_figure=False,
                close_plot=False,
                show_plot=False,
                fig_handle=fig,
                show_colorbar=False,
                show_triangulation=False,
                show_axis=False,
                show_boundary=False,
                clim=(global_min, global_max),
            )

        # plot mesh edges
        for e in self.solver.glob_fun_sp.mesh.edges:
            plt.plot(e.x[0, :], e.x[1, :], "k")

        if fill and show_colorbar:
            sm = ScalarMappable(
                norm=plt.Normalize(vmin=global_min, vmax=global_max),
                cmap=colormap,
            )
            plt.colorbar(
                mappable=sm,
                ax=plt.gca(),
                fraction=0.046,
                pad=0.04,
            )

        plt.axis("equal")
        plt.gca().set_aspect("equal")
        plt.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0
        )
        if not show_axis:
            plt.axis("off")

        if len(title) > 0:
            plt.title(title)

        if save_fig:
            save_figure(filename)

        if show_plot:
            plt.show()

        if show_plot or save_fig:
            plt.close()


def _range_on_cell(
    vals: np.ndarray, current_min: float, current_max: float
) -> tuple[float, float]:
    local_min = np.min(vals)
    local_max = np.max(vals)
    if local_min < current_min:
        current_min = local_min
    if local_max > current_max:
        current_max = local_max
    return current_min, current_max
