"""
Plot a local function.

Classes
-------
LocalFunctionPlot

Functions
---------
_construct_gradient_component_on_boundary
_get_hole_mask
_point_is_inside
_point_is_inside_simple
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..locfun.local_poisson import LocalPoissonFunction
from .plot_util import get_axis_limits, get_figure_size, save_figure

TOL = 1e-8
MAX_GRAD = 1e3


class LocalFunctionPlot:
    """
    Plot a local function.

    Attributes
    ----------
    v : LocalPoissonFunction
        The local function to be plotted.
    triangulation : tri.Triangulation
        A triangulation of the interior points together with the boundary
        points.
    hole_mask : np.ndarray
        A mask that removes triangles that lie in a hole.
    use_interp : bool
        If True, the triangulation includes boundary points.
    """

    v: LocalPoissonFunction
    use_interp: bool

    def __init__(
        self, v: LocalPoissonFunction, use_interp: bool = True
    ) -> None:
        """
        Initialize a LocalFunctionPlot object.

        Parameters
        ----------
        v : LocalPoissonFunction
            The local function to be plotted.
        """
        self.set_local_function(v)
        self._set_use_interp(use_interp)

    def set_local_function(self, v: LocalPoissonFunction) -> None:
        """
        Set the local function to be plotted.

        Parameters
        ----------
        v : LocalPoissonFunction
            The local function to be plotted.
        """
        if not isinstance(v, LocalPoissonFunction):
            raise TypeError("v must be a LocalPoissonFunction")
        self.v = v

    def _set_use_interp(self, use_interp: bool) -> None:
        if not isinstance(use_interp, bool):
            raise TypeError("use_interp must be a boolean")
        self.use_interp = use_interp

    def draw(
        self,
        plot_type: str = "values",
        show_plot: bool = True,
        filename: str = "",
        **kwargs: Any
    ) -> None:
        """
        Draw the plot of the internal values.

        Parameters
        ----------
        plot_type : str
            The type of plot. Must be one of "values", "grad_x1", "grad_x2", or
            "grad_norm".
        show_plot : bool
            If True, the plot is shown.
        filename : str
            If not empty, the plot is saved to this file.

        Other Parameters
        ----------------
        fill : bool
            If True, a heatmap is plotted. If False, a contour plot is plotted.
            Default is True.
        title : str
            The title of the plot. Default is "", i.e. no title.
        levels : int
            The number of levels in the contour plot. Default is 32.
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        show_colorbar : bool
            If True, a colorbar is shown. Default is True.
        show_axis : bool
            If True, the axis is shown. Default is True.
        show_boundary : bool
            If True, the boundary is shown. Default is True.
        show_triangulation : bool
            If True, the triangulation is shown. Default is False.
        use_log10 : bool
            If True, the log10 function is applied to interior and boundary
            values. Default is False.
        """
        if plot_type == "values":
            self._draw_vals(show_plot=show_plot, filename=filename, **kwargs)
        elif plot_type == "grad_x1":
            self._draw_grad_x1(show_plot=show_plot, filename=filename, **kwargs)
        elif plot_type == "grad_x2":
            self._draw_grad_x2(show_plot=show_plot, filename=filename, **kwargs)
        elif plot_type == "grad_norm":
            self._draw_grad_norm(
                show_plot=show_plot, filename=filename, **kwargs
            )
        else:
            raise ValueError(
                "type must be 'values', 'grad_x1', 'grad_x2', or 'grad_norm'"
            )

    def _draw_vals(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        if self.use_interp:
            bdry_vals = self.v.harm.trace.values + self.v.poly.trace.values
        else:
            bdry_vals = None
        self._draw_generic(
            interior_values=self.v.int_vals,
            boundary_values=bdry_vals,
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )

    def _draw_grad_x1(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        if self.use_interp:
            grad_x1 = _construct_gradient_on_boundary_x1(self.v)
        else:
            grad_x1 = None
        self._draw_generic(
            interior_values=self.v.int_grad1,
            boundary_values=grad_x1,
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )

    def _draw_grad_x2(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        if self.use_interp:
            grad_x2 = _construct_gradient_on_boundary_x2(self.v)
        else:
            grad_x2 = None
        self._draw_generic(
            interior_values=self.v.int_grad2,
            boundary_values=grad_x2,
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )

    def _draw_grad_norm(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        if self.v.int_grad1 is None or self.v.int_grad2 is None:
            raise ValueError("int_grad1 and int_grad2 must be set")
        if self.use_interp:
            grad_x1 = _construct_gradient_on_boundary_x1(self.v)
            grad_x2 = _construct_gradient_on_boundary_x2(self.v)
            grad_norm = np.sqrt(grad_x1**2 + grad_x2**2)
        else:
            grad_norm = None
        grad_norm_interior = np.sqrt(self.v.int_grad1**2 + self.v.int_grad2**2)
        self._draw_generic(
            interior_values=grad_norm_interior,
            boundary_values=grad_norm,
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )

    def _draw_generic(
        self,
        interior_values: Optional[np.ndarray],
        boundary_values: Optional[np.ndarray],
        show_plot: bool = True,
        filename: str = "",
        **kwargs: Any
    ) -> None:
        # unpack kwargs
        fill = kwargs.get("fill", True)
        title = kwargs.get("title", "")
        levels = kwargs.get("levels", 32)
        colormap = kwargs.get("colormap", None)
        show_colorbar = kwargs.get("show_colorbar", True)
        show_axis = kwargs.get("show_axis", True)
        show_boundary = kwargs.get("show_boundary", True)
        show_triangulation = kwargs.get("show_triangulation", False)
        close_plot = kwargs.get("close_plot", True)
        new_figure = kwargs.get("new_figure", True)
        fig_handle = kwargs.get("fig_handle", None)
        clim = kwargs.get("clim", None)
        use_log10 = kwargs.get("use_log10", False)

        # create new figure if necessary
        if new_figure and fig_handle is None:
            min_x, max_x, min_y, max_y = get_axis_limits(
                self.v.mesh_cell.get_edges()
            )
            w, h = get_figure_size(min_x, max_x, min_y, max_y)
            fig_handle = plt.figure(figsize=(w, h))

        # use the provided figure handle
        if fig_handle is not None:
            plt.figure(fig_handle.number)

        # concatenate interior and boundary values
        if interior_values is None:
            raise NotImplementedError(
                "Skeleton plotting is not yet implemented."
            )
        if self.use_interp and boundary_values is not None:
            vals = np.concatenate((interior_values, boundary_values))
        else:
            vals = interior_values

        # apply log10
        if use_log10:
            vals[vals < 1e-16] = 1e-16
            vals = np.log10(vals)

        # plot values
        self._apply_masks_and_draw(vals, fill, levels, clim)

        # plot mesh
        if show_triangulation:
            plt.triplot(self.v.mesh_cell.triangulation, "-k")

        # axis and shape
        plt.axis("equal")
        plt.gca().set_aspect("equal")
        plt.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0
        )
        if not show_axis:
            plt.axis("off")

        # plot boundary
        if show_boundary:
            self._plot_edges()

        # title, colormap, colorbar, axis
        self._make_plot_extras(title, colormap, show_colorbar)

        # save or show plot
        self._output_plot(show_plot, filename)

        # close plot
        if close_plot:
            plt.close()

    def _apply_masks_and_draw(
        self,
        vals: np.ndarray,
        fill: bool,
        levels: int,
        clim: Optional[tuple[float, float]],
    ) -> None:

        # find inf and nan values
        pointwise_mask = np.logical_not(np.isfinite(vals))

        # get only the triangles that have no bad points
        inf_mask = np.any(
            pointwise_mask[self.v.mesh_cell.triangulation.triangles], axis=1
        )

        # remove triangles from holes and with bad points
        mask = np.logical_or(self.v.mesh_cell.hole_mask, inf_mask)

        # optionally remove boundary triangles
        if not self.use_interp:
            mask[self.v.mesh_cell.boundary_mask] = True

        # apply mask
        self.v.mesh_cell.triangulation.set_mask(mask)

        # plot interior values
        if fill:
            contour = plt.tricontourf(
                self.v.mesh_cell.triangulation, vals, levels=levels
            )
        else:
            contour = plt.tricontour(
                self.v.mesh_cell.triangulation, vals, levels=levels
            )

        # set color limits
        if clim is not None:
            contour.set_clim(clim)

    def _plot_edges(self) -> None:
        for edge in self.v.mesh_cell.get_edges():
            plt.plot(*edge.get_sampled_points(ignore_endpoint=False), "-k")

    def _make_plot_extras(
        self, title: str, colormap: Optional[str], show_colorbar: bool
    ) -> None:

        # set title
        if title:
            plt.title(title)

        # set colormap
        if colormap is not None:
            plt.set_cmap(colormap)

        # set colorbar
        if show_colorbar:
            divider = make_axes_locatable(plt.gca())
            colorbar_axes = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cax=colorbar_axes, mappable=plt.gci())

    def _output_plot(self, show_plot: bool, filename: str) -> None:
        if filename:
            save_figure(filename)
        if show_plot:
            plt.show()


def _construct_gradient_on_boundary_x1(
    v: LocalPoissonFunction,
) -> np.ndarray:
    return _construct_gradient_component_on_boundary(v, 1)


def _construct_gradient_on_boundary_x2(
    v: LocalPoissonFunction,
) -> np.ndarray:
    return _construct_gradient_component_on_boundary(v, 2)


def _construct_gradient_component_on_boundary(
    v: LocalPoissonFunction, component: int
) -> np.ndarray:
    if v.harm.trace.w_norm_deriv is None:
        raise ValueError("harm.w_norm_deriv must be set")
    if v.harm.trace.w_tang_deriv is None:
        raise ValueError("harm.w_tang_deriv must be set")
    if v.poly.trace.w_norm_deriv is None:
        raise ValueError("poly.w_norm_deriv must be set")
    if v.poly.trace.w_tang_deriv is None:
        raise ValueError("poly.w_tang_deriv must be set")
    num_pts = v.mesh_cell.num_pts
    if component == 1:
        b1 = np.ones((num_pts,))
        b2 = np.zeros((num_pts,))
    elif component == 2:
        b1 = np.zeros((num_pts,))
        b2 = np.ones((num_pts,))
    else:
        raise ValueError("component must be 1 or 2")
    norm_comp = v.mesh_cell.dot_with_normal(b1, b2)
    tang_comp = v.mesh_cell.dot_with_tangent(b1, b2)
    wnd = v.harm.trace.w_norm_deriv + v.poly.trace.w_norm_deriv
    wtd = v.harm.trace.w_tang_deriv + v.poly.trace.w_tang_deriv
    weighted_grad = norm_comp * wnd + tang_comp * wtd
    wgt = v.mesh_cell.get_dx_norm()
    error_settings = np.seterr(divide="ignore")
    grad_xj = weighted_grad / wgt
    np.seterr(**error_settings)
    grad_xj[wgt < TOL] = np.nan
    grad_xj[grad_xj > MAX_GRAD] = np.nan
    return grad_xj
