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
from matplotlib import tri
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..locfun.local_poisson import LocalPoissonFunction
from .plot_util import save_figure

TOL = 1e-8


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
    triangulation: tri.Triangulation
    hole_mask: np.ndarray
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
        self._build_triangulation()

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

    def _build_triangulation(self) -> None:
        # interior points
        x1 = self.v.mesh_cell.int_x1
        x2 = self.v.mesh_cell.int_x2

        # concatenate interior points with boundary points
        if self.use_interp:
            y1, y2 = self.v.mesh_cell.get_boundary_points()
            x1 = np.concatenate((x1, y1))
            x2 = np.concatenate((x2, y2))

        # triangulate
        crude_triangulation = tri.Triangulation(x1, x2)

        # partition boundary into outer and hole edges
        outer_x, outer_y = self.v.mesh_cell.components[0].get_sampled_points()
        hole_x = []
        hole_y = []
        for component in self.v.mesh_cell.components[1:]:
            x, y = component.get_sampled_points()
            hole_x.append(x)
            hole_y.append(y)

        # remove triangles with all three vertices on a hole edge
        self.hole_mask = _get_hole_mask(
            crude_triangulation, outer_x, outer_y, hole_x, hole_y, radius=1e-6
        )

        # set triangulation
        self.triangulation = tri.Triangulation(
            x1, x2, triangles=crude_triangulation.triangles, mask=self.hole_mask
        )

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
            The type of plot. Must be one of "values", "grad_x1", "grad_x2",
            or "grad_norm".
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
        bdry_vals = self.v.harm.trace.values + self.v.poly.trace.values
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
        grad_x1 = _construct_gradient_on_boundary_x1(self.v)
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
        grad_x2 = _construct_gradient_on_boundary_x2(self.v)
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
        grad_x1 = _construct_gradient_on_boundary_x1(self.v)
        grad_x2 = _construct_gradient_on_boundary_x2(self.v)
        grad_norm = np.sqrt(grad_x1**2 + grad_x2**2)
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
        interior_values: np.ndarray,
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

        # create new figure if necessary
        if new_figure and fig_handle is None:
            fig_handle = plt.figure()

        # use the provided figure handle
        if fig_handle is not None:
            plt.figure(fig_handle.number)

        # concatenate interior and boundary values
        if self.use_interp and boundary_values is not None:
            vals = np.concatenate((interior_values, boundary_values))
        else:
            vals = interior_values

        # plot values
        self._apply_masks_and_draw(vals, fill, levels)

        # plot mesh
        if show_triangulation:
            plt.triplot(self.triangulation, "-k")

        # axis and shape
        plt.axis("equal")
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
        self, vals: np.ndarray, fill: bool, levels: int
    ) -> None:

        # find inf and nan values
        pointwise_mask = np.logical_not(np.isfinite(vals))

        # get only the triangles that have no bad points
        inf_mask = np.any(pointwise_mask[self.triangulation.triangles], axis=1)

        # remove triangles from holes and with bad points
        mask = np.logical_or(self.hole_mask, inf_mask)

        # apply mask
        self.triangulation.set_mask(mask)

        # plot interior values
        if fill:
            plt.tricontourf(self.triangulation, vals, levels=levels)
        else:
            plt.tricontour(self.triangulation, vals, levels=levels)

    def _plot_edges(self) -> None:
        for edge in self.v.mesh_cell.get_edges():
            plt.plot(*edge.get_sampled_points(), "-k")

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
    return grad_xj


def _get_hole_mask(
    triangulation: tri.Triangulation,
    outer_x: np.ndarray,
    outer_y: np.ndarray,
    holes_x: list[np.ndarray],
    holes_y: list[np.ndarray],
    radius: float,
) -> np.ndarray:
    # Test if the midpoint of each edge lies inside the domain. If it lies
    # outside the domain, the edge is removed.
    mask = np.zeros(triangulation.triangles.shape[0], dtype=bool)
    for t in range(triangulation.triangles.shape[0]):
        for i in range(3):
            if mask[t]:
                continue
            a = triangulation.triangles[t, i]
            b = triangulation.triangles[t, (i + 1) % 3]

            # if both points lie consecutively on the boundary, the edge is kept
            if not _edge_is_on_boundary(
                [triangulation.x[a], triangulation.x[b]],
                [triangulation.y[a], triangulation.y[b]],
                outer_x,
                outer_y,
                holes_x,
                holes_y,
                radius,
            ):
                mid_x = 0.5 * (triangulation.x[a] + triangulation.x[b])
                mid_y = 0.5 * (triangulation.y[a] + triangulation.y[b])
                mask[t] = not _point_is_inside(
                    mid_x, mid_y, outer_x, outer_y, holes_x, holes_y, radius
                )
    return mask

def _edge_is_on_boundary(
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    outer_x: np.ndarray,
    outer_y: np.ndarray,
    holes_x: list[np.ndarray],
    holes_y: list[np.ndarray],
    radius: float,
) -> bool:
    # test if the edge is any of the boundary edges
    if _edge_is_on_boundary_simple(edge_x, edge_y, outer_x, outer_y, radius):
        return True
    for hole_x, hole_y in zip(holes_x, holes_y):
        if _edge_is_on_boundary_simple(edge_x, edge_y, hole_x, hole_y, radius):
            return True
    return False

def _edge_is_on_boundary_simple(
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    path_x: np.ndarray,
    path_y: np.ndarray,
    radius: float,
) -> bool:
    # return true if the edge is on the boundary
    # there are two cases: when the edge is parallel to the boundary and when
    # the edge is oppositely oriented to the boundary
    n = len(path_x)
    for i in range(n):
        x1, y1 = path_x[i], path_y[i]
        x2, y2 = path_x[(i + 1) % n], path_y[(i + 1) % n]
        if _edge_is_on_boundary_segment(edge_x, edge_y, x1, y1, x2, y2, radius):
            return True
    return False

def _edge_is_on_boundary_segment(
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
) -> bool:
    a = np.array([x2 - x1, y2 - y1])
    b = np.array([edge_x[1] - edge_x[0], edge_y[1] - edge_y[0]])
    if np.linalg.norm(a - b) < radius:
        return True
    if np.linalg.norm(a + b) < radius:
        return True
    return False

def _point_is_inside(
    point_x: float,
    point_y: float,
    outer_x: np.ndarray,
    outer_y: np.ndarray,
    holes_x: list[np.ndarray],
    holes_y: list[np.ndarray],
    radius: float,
) -> bool:
    if not _point_is_inside_simple(point_x, point_y, outer_x, outer_y, radius):
        return False
    for hole_x, hole_y in zip(holes_x, holes_y):
        if _point_is_inside_simple(point_x, point_y, hole_x, hole_y, radius):
            return False
    return True


def _point_is_inside_simple(
    point_x: float,
    point_y: float,
    path_x: np.ndarray,
    path_y: np.ndarray,
    radius: float,
) -> bool:
    polygon = np.zeros((len(path_x), 2))
    polygon[:, 0] = np.array(path_x)
    polygon[:, 1] = np.array(path_y)
    return Path(polygon).contains_point((point_x, point_y), radius=radius)
