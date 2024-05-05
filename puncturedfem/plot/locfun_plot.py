"""
Plot a local function.

Classes
-------
LocalFunctionPlot
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
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
    colormap : Optional[Colormap]
        The colormap used for the plot.
    """

    v: LocalFunction
    fill: bool
    title: str
    levels: int
    show_colorbar: bool
    show_axis: bool
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
        self.colormap = kwargs.get("colormap", None)

    def _draw_generic(
        self,
        vals: np.ndarray,
        show_plot: bool = True,
        filename: str = "",
        **kwargs: Any
    ) -> None:
        self._unpack_kwargs(kwargs)
        edges = self.v.nyst.K.get_edges()
        MeshPlot(edges).draw(show_plot=False, keep_open=True)
        x1 = self.v.nyst.K.int_x1
        x2 = self.v.nyst.K.int_x2
        vals = self.v.int_vals
        if self.fill:
            plt.contourf(x1, x2, vals, levels=self.levels, cmap=self.colormap)
        else:
            plt.contour(x1, x2, vals, levels=self.levels, cmap=self.colormap)
        if self.show_colorbar:
            divider = make_axes_locatable(plt.gca())
            colorbar_axes = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cax=colorbar_axes, mappable=plt.gci())
        if self.title:
            plt.title(self.title)
        if not self.show_axis:
            plt.axis("off")
        if filename:
            save_figure(filename)
        if show_plot:
            plt.show()
        plt.close()

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
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        """
        self._draw_generic(
            vals=self.v.int_vals,
            show_plot=show_plot,
            filename=filename,
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
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        """
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
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        """
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
        colormap : Optional[Colormap]
            The colormap used for the plot. Default is None.
        """
        self._draw_generic(
            vals=np.sqrt(self.v.int_grad1**2 + self.v.int_grad2**2),
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )
