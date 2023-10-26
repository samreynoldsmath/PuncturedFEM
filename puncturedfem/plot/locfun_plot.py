"""
plot_global_solution.py
=======================

Module for plotting the global solution.
"""

import matplotlib.pyplot as plt

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
        self.v = v

    def draw(
        self,
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
