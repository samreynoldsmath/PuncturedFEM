"""
mesh_plot.py
============

Module containing the MeshPlot class, which is used to plot edges, cells, and
meshes.
"""

from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..mesh.edge import Edge
from ..mesh.quad import QuadDict, get_quad_dict
from .plot_util import get_axis_limits, get_figure_size, save_figure


class MeshPlot:
    """
    Class for plotting lists of edges, such as cell boundaries and meshes.
    """

    edges: list[Edge]
    quad_dict: QuadDict
    title: str
    show_orientation: bool
    show_grid: bool
    show_axis: bool
    keep_open: bool
    pad: float

    def __init__(
        self, edges: list[Edge], n: int = 32, reparameterize: bool = False
    ) -> None:
        """
        Constructor for MeshPlot class.

        Parameters
        ----------
        edges : list of Edge
            The edges to be plotted
        """
        self.quad_dict = get_quad_dict(n)
        self.set_edges(deepcopy(edges))
        for e in self.edges:
            if reparameterize or not e.is_parameterized:
                e.parameterize(self.quad_dict)

    def _unpack_kwargs(self, kwargs: dict[str, Any]) -> None:
        """
        Unpack the keyword arguments.
        """
        self.title = kwargs.get("title", "")
        self.show_orientation = kwargs.get("show_orientation", False)
        self.show_grid = kwargs.get("show_grid", False)
        self.show_axis = kwargs.get("show_axis", True)
        self.keep_open = kwargs.get("keep_open", False)
        self.pad = kwargs.get("pad", 0.1)

    def draw(
        self, show_plot: bool = True, filename: str = "", **kwargs: Any
    ) -> None:
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
        self._unpack_kwargs(kwargs)

        # determine axes and figure size
        min_x, max_x, min_y, max_y = get_axis_limits(self.edges, self.pad)
        w, h = get_figure_size(min_x, max_x, min_y, max_y)

        # create figure
        plt.figure(figsize=(w, h))
        self._plot_edges()
        plt.axis((min_x, max_x, min_y, max_y))
        plt.grid(self.show_grid)
        if self.title:
            plt.title(self.title)
        if not self.show_axis:
            plt.axis("off")
        if filename:
            save_figure(filename)
        if show_plot:
            plt.show()
        if not self.keep_open:
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
