"""
plot_global_solution.py
=======================

Module for plotting the global solution.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..locfun.locfun import LocalFunction


class LocalFunctionPlot:
    """
    Class for plotting a local function.
    """

    v: LocalFunction
    fill: bool
    title: str

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

    def _draw_generic(
        self,
        vals: np.ndarray,
        show_plot: bool = True,
        filename: str = "",
        fill: bool = True,
        title: str = "",
        levels: int = 32,
        show_colorbar: bool = True,
    ) -> None:
        """
        Draw the plot.
        """
        plt.figure()
        edges = self.v.nyst.K.get_edges()
        for e in edges:
            plt.plot(e.x[0, :], e.x[1, :], "k")
        x1 = self.v.nyst.K.int_x1
        x2 = self.v.nyst.K.int_x2
        vals = self.v.int_vals
        if fill:
            plt.contourf(x1, x2, vals, levels=levels)
        else:
            plt.contour(x1, x2, vals, levels=levels)
        if show_colorbar:
            plt.colorbar()
        plt.axis("equal")
        if title:
            plt.title(title)
        if show_plot:
            plt.show()
        if filename:
            plt.savefig(filename)

    def draw_vals(
        self,
        show_plot: bool = True,
        filename: str = "",
        fill: bool = True,
        title: str = "",
        levels: int = 32,
        show_colorbar: bool = True,
    ) -> None:
        """
        Draw the plot of the internal values.
        """
        self._draw_generic(
            vals=self.v.int_vals,
            show_plot=show_plot,
            filename=filename,
            fill=fill,
            title=title,
            levels=levels,
            show_colorbar=show_colorbar,
        )

    def draw_grad_x1(
        self,
        show_plot: bool = True,
        filename: str = "",
        fill: bool = True,
        title: str = "",
        levels: int = 32,
        show_colorbar: bool = True,
    ) -> None:
        """
        Draw the plot of the x1 component of the gradient.
        """
        self._draw_generic(
            vals=self.v.int_grad1,
            show_plot=show_plot,
            filename=filename,
            fill=fill,
            title=title,
            levels=levels,
            show_colorbar=show_colorbar,
        )

    def draw_grad_x2(
        self,
        show_plot: bool = True,
        filename: str = "",
        fill: bool = True,
        title: str = "",
        levels: int = 32,
        show_colorbar: bool = True,
    ) -> None:
        """
        Draw the plot of the x2 component of the gradient.
        """
        self._draw_generic(
            vals=self.v.int_grad2,
            show_plot=show_plot,
            filename=filename,
            fill=fill,
            title=title,
            levels=levels,
            show_colorbar=show_colorbar,
        )

    def draw_grad_norm(
        self,
        show_plot: bool = True,
        filename: str = "",
        fill: bool = True,
        title: str = "",
        levels: int = 32,
        show_colorbar: bool = True,
    ) -> None:
        """
        Draw the plot of the norm of the gradient.
        """
        self._draw_generic(
            vals=np.sqrt(self.v.int_grad1**2 + self.v.int_grad2**2),
            show_plot=show_plot,
            filename=filename,
            fill=fill,
            title=title,
            levels=levels,
            show_colorbar=show_colorbar,
        )
