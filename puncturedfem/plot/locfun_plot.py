"""
Plot a local function.

Classes
-------
LocalFunctionPlot

Functions
---------
_remove_holes
_point_is_inside
_point_is_inside_simple
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri
from matplotlib.colors import Colormap
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..locfun.locfun import LocalFunction
from .mesh_plot import MeshPlot
from .plot_util import save_figure


class LocalFunctionPlot:
    """
    Plot a local function.

    Attributes
    ----------
    v : LocalFunction
        The local function to be plotted.
    fill : bool
        If True, a heatmap is plotted. If False, a contour plot is plotted.
    title : str
        The title of the plot.
    levels : int
        The number of levels in the contour plot.
    show_colorbar : bool
        If True, a colorbar is shown.
    show_axis : bool
        If True, the axis is shown.
    use_interp : bool
        If True, values near the boundary are interpolated.
    colormap : Optional[Colormap]
        The colormap used for the plot.
    """

    v: LocalFunction
    fill: bool
    title: str
    levels: int
    show_colorbar: bool
    show_axis: bool
    use_interp: bool
    colormap: Optional[Colormap]

    def __init__(self, v: LocalFunction) -> None:
        """
        Initialize a LocalFunctionPlot object.

        Parameters
        ----------
        v : LocalFunction
            The local function to be plotted.
        """
        self.set_local_function(v)

    def set_local_function(self, v: LocalFunction) -> None:
        """
        Set the local function to be plotted.

        Parameters
        ----------
        v : LocalFunction
            The local function to be plotted.
        """
        if not isinstance(v, LocalFunction):
            raise TypeError("v must be a LocalFunction")
        self.v = v

    def _unpack_kwargs(self, kwargs: dict) -> None:
        self.fill = kwargs.get("fill", True)
        self.title = kwargs.get("title", "")
        self.levels = kwargs.get("levels", 32)
        self.show_colorbar = kwargs.get("show_colorbar", True)
        self.show_axis = kwargs.get("show_axis", True)
        self.use_interp = kwargs.get("use_interp", True)
        self.colormap = kwargs.get("colormap", None)

    def _draw_generic(
        self,
        interior_values: np.ndarray,
        show_plot: bool = True,
        filename: str = "",
        boundary_values: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> None:
        self._unpack_kwargs(kwargs)
        edges = self.v.nyst.K.get_edges()
        MeshPlot(edges).draw(show_plot=False, keep_open=True)
        if self.use_interp:
            if boundary_values is None:
                raise ValueError("boundary_values must be provided")
            self._draw_interp(interior_values, boundary_values)
        else:
            self._draw_classic(interior_values)
        if self.title:
            plt.title(self.title)
        if not self.show_axis:
            plt.axis("off")
        if filename:
            save_figure(filename)
        if show_plot:
            plt.show()
        plt.close()

    def _draw_classic(self, interior_values: np.ndarray) -> None:
        x1 = self.v.nyst.K.int_x1
        x2 = self.v.nyst.K.int_x2
        if self.fill:
            plt.contourf(
                x1, x2, interior_values, levels=self.levels, cmap=self.colormap
            )
        else:
            plt.contour(
                x1, x2, interior_values, levels=self.levels, cmap=self.colormap
            )
        if self.show_colorbar:
            divider = make_axes_locatable(plt.gca())
            colorbar_axes = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cax=colorbar_axes, mappable=plt.gci())

    def _draw_interp(
        self, interior_values: np.ndarray, boundary_values: np.ndarray
    ) -> None:
        # interior points and values
        x1 = list(self.v.nyst.K.int_x1.flatten())
        x2 = list(self.v.nyst.K.int_x2.flatten())
        vals = list(interior_values.flatten())

        # delete nan values
        nan_indices = np.isnan(vals)
        for i in reversed(range(len(vals))):
            if nan_indices[i]:
                del x1[i]
                del x2[i]
                del vals[i]

        # boundary points
        y1, y2 = self.v.nyst.K.get_boundary_points()

        # concatenate all points and values
        x1.extend(list(y1))
        x2.extend(list(y2))
        vals.extend(list(boundary_values))

        # cast to numpy arrays
        X1 = np.array(x1)
        X2 = np.array(x2)
        V = np.array(vals)

        # triangulate
        triangulation = tri.Triangulation(X1, X2)

        # partition boundary into outer and hole edges
        outer_x, outer_y = self.v.nyst.K.components[0].get_sampled_points()
        hole_x = []
        hole_y = []
        for component in self.v.nyst.K.components[1:]:
            x, y = component.get_sampled_points()
            hole_x.append(x)
            hole_y.append(y)

        # remove triangles with all three vertices on a hole edge
        triangulation = _remove_holes(
            triangulation, outer_x, outer_y, hole_x, hole_y
        )

        # plot
        if self.fill:
            plt.tricontourf(triangulation, V)
            # plt.triplot(triangulation, "-k")
        else:
            plt.tricontour(triangulation, V)

    def draw(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        """
        Draw the plot of the internal values.

        Parameters
        ----------
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
        show_colorbar : bool
            If True, a colorbar is shown. Default is True.
        show_axis : bool
            If True, the axis is shown. Default is True.
        use_interp : bool
            If True, values near the boundary are interpolated. Default is
            False.
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        """
        self.draw_vals(show_plot=show_plot, filename=filename, **kwargs)

    def draw_vals(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        """
        Draw the plot of the internal values.

        Parameters
        ----------
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
        show_colorbar : bool
            If True, a colorbar is shown. Default is True.
        show_axis : bool
            If True, the axis is shown. Default is True.
        use_interp : bool
            If True, values near the boundary are interpolated. Default is
            False.
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        """
        self._draw_generic(
            interior_values=self.v.int_vals,
            show_plot=show_plot,
            filename=filename,
            boundary_values=self.v.get_trace_values(),
            **kwargs
        )

    def draw_grad_x1(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        """
        Draw the plot of the x1 component of the gradient.

        Parameters
        ----------
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
        show_colorbar : bool
            If True, a colorbar is shown. Default is True.
        show_axis : bool
            If True, the axis is shown. Default is True.
        use_interp : bool
            If True, values near the boundary are interpolated. Default is
            False.
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        """
        if self.use_interp:
            raise NotImplementedError(
                "Interpolation not implemented for gradients"
            )
        self._draw_generic(
            vals=self.v.int_grad1,
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )

    def draw_grad_x2(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        """
        Draw the plot of the x2 component of the gradient.

        Parameters
        ----------
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
        show_colorbar : bool
            If True, a colorbar is shown. Default is True.
        show_axis : bool
            If True, the axis is shown. Default is True.
        use_interp : bool
            If True, values near the boundary are interpolated. Default is
            False.
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        """
        if self.use_interp:
            raise NotImplementedError(
                "Interpolation not implemented for gradients"
            )
        self._draw_generic(
            vals=self.v.int_grad2,
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )

    def draw_grad_norm(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        """
        Draw the plot of the norm of the gradient.

        Parameters
        ----------
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
        show_colorbar : bool
            If True, a colorbar is shown. Default is True.
        show_axis : bool
            If True, the axis is shown. Default is True.
        use_interp : bool
            If True, values near the boundary are interpolated. Default is
            False.
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        """
        if self.use_interp:
            raise NotImplementedError(
                "Interpolation not implemented for gradients"
            )
        self._draw_generic(
            vals=np.sqrt(self.v.int_grad1**2 + self.v.int_grad2**2),
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )


def _remove_holes(
    triangulation: tri.Triangulation,
    outer_x: np.ndarray,
    outer_y: np.ndarray,
    holes_x: list[np.ndarray],
    holes_y: list[np.ndarray],
    radius: float = 1e-8,
) -> tri.Triangulation:
    """
    The midpoint of each edge is tested to see if it lies inside the domain.
    If it lies outside the domain, the edge is removed.
    """
    mask = np.zeros(triangulation.triangles.shape[0], dtype=bool)
    for t in range(triangulation.triangles.shape[0]):
        for i in range(3):
            if mask[t]:
                continue
            a = triangulation.triangles[t, i]
            b = triangulation.triangles[t, (i + 1) % 3]
            mid_x = 0.5 * (triangulation.x[a] + triangulation.x[b])
            mid_y = 0.5 * (triangulation.y[a] + triangulation.y[b])
            mask[t] = not _point_is_inside(
                mid_x, mid_y, outer_x, outer_y, holes_x, holes_y, radius
            )
    triangulation.set_mask(mask)
    return triangulation


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
