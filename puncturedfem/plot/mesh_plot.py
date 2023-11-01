"""
mesh_plot.py
============

Module containing the MeshPlot class, which is used to plot edges, cells, and
meshes.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..mesh.edge import Edge


class MeshPlot:
    """
    Class for plotting lists of edges, such as cell boundaries and meshes.
    """

    def __init__(
        self,
        edges: list[Edge],
        show_orientation: bool = False,
        show_grid: bool = False,
        title: str = "",
    ) -> None:
        """
        Constructor for MeshPlot class.

        Parameters
        ----------
        edges : list of Edge
            The edges to be plotted
        show_orientation : bool, optional
            Whether to plot the orientation of the edges. The default is False.
        show_grid : bool, optional
            Whether to plot a grid. The default is False.
        title : str, optional
            The title of the plot. The default is "", which results in no title.
        """
        self.set_edges(edges)
        self.title = title
        self.show_orientation = show_orientation
        self.show_grid = show_grid

    def draw(self, show_plot: bool = True, filename: str = "") -> None:
        """
        Creates the plot.

        Parameters
        ----------
        show_plot : bool, optional
            Whether to show the plot. The default is True.
        filename : str, optional
            The filename to save the plot to. The default is "", which results
            in no file being saved.
        """
        plt.figure()
        self._plot_edges()
        plt.axis("equal")
        plt.grid(self.show_grid)
        if self.title:
            plt.title(self.title)
        if filename:
            plt.savefig(filename)
        if show_plot:
            plt.show()
        plt.close()

    def set_edges(self, edges: list[Edge]) -> None:
        """
        Sets the edges to be plotted.
        """
        self._validate_edges(edges)
        self.edges = edges

    def _validate_edges(self, edges: list[Edge]) -> None:
        if not isinstance(edges, list):
            raise TypeError("edges must be a list of Edge objects")
        for edge in edges:
            if not isinstance(edge, Edge):
                raise TypeError("edges must be a list of Edge objects")

    def _plot_edges(self) -> None:
        for edge in self.edges:
            self._plot_edge(edge, self.show_orientation)

    def _plot_edge(self, edge: Edge, show_orientation: bool) -> None:
        x, y = edge.get_sampled_points()
        if show_orientation:
            self._plot_oriented_edge_points(x, y)
        else:
            plt.plot(x, y, "-k")

    def _plot_oriented_edge_points(self, x: np.ndarray, y: np.ndarray) -> None:
        u = np.roll(x, -1) - x
        v = np.roll(y, -1) - y
        plt.quiver(
            x[:-1],
            y[:-1],
            u[:-1],
            v[:-1],
            scale=1,
            angles="xy",
            scale_units="xy",
        )
