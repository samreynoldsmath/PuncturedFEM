"""
plot_global_solution.py
=======================

Module for plotting the global solution.
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
    Class for plotting a local function.
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
        Constructor for the LocalFunctionPlot class.

        Parameters
        ----------
        v : LocalFunction
            The local function to be plotted.
        """
        self.set_local_function(v)

    def set_local_function(self, v: LocalFunction) -> None:
        """
        Set the local function to be plotted.
        """
        if not isinstance(v, LocalFunction):
            raise TypeError("v must be a LocalFunction")
        self.v = v

    def _unpack_kwargs(self, kwargs: dict) -> None:
        """
        Unpack the keyword arguments.
        """
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
        """
        Draw the plot.
        """
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
        Draw the plot of the internal values. (Alias for draw_vals.)
        """
        self.draw_vals(show_plot=show_plot, filename=filename, **kwargs)

    def draw_vals(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
        """
        Draw the plot of the internal values.
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
        """
        self._draw_generic(
            vals=np.sqrt(self.v.int_grad1**2 + self.v.int_grad2**2),
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )
