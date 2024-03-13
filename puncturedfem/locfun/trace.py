"""
trace.py
========

This module contains the class `DirichletTrace` which is used to represent the
trace of a LocalFunction on the boundary of a MeshCell.

TODO: Deprecate the PiecewisePolynomial class
"""

from typing import Union

import numpy as np

from ..mesh.cell import MeshCell
from ..mesh.edge import Edge
from ..util.types import Func_R2_R, is_Func_R2_R
from .poly.poly import Polynomial


class DirichletTrace:
    """
    This class is used to represent the trace of a LocalFunction on the
    boundary of a MeshCell. The trace is represented as a list of edges, and
    the values of the trace on the edges are stored in an array. The class also
    contains methods to set the values of the trace and to evaluate the trace
    on the edges.

    Attributes
    ----------
    edges : list[Edge]
        The edges on which the trace is defined.
    num_pts : int
        The number of points on the edges.
    values : np.ndarray
        The values of the trace on the edges.
    funcs : list[Func_R2_R]
        The functions used to define the trace.
    edge_sampled_indices : list[tuple[int, int]]
        The start and end indices of the edges in the values array.

    Usage
    -----
    TODO: Add usage examples
    """

    edges: list[Edge]
    num_edges: int
    num_pts: int
    values: np.ndarray
    funcs: list[Func_R2_R]
    edge_sampled_indices: list[tuple[int, int]]

    def __init__(
        self, edges: Union[MeshCell, list[Edge]], custom: bool = False
    ) -> None:
        """
        Constructor for the DirichletTrace class.

        Parameters
        ----------
        edges : Union[MeshCell, list[Edge]]
            The edges on which the trace is defined.
        custom : bool, optional
            A flag indicating whether the trace is defined
            using custom functions. If True, the user must
            set the values and functions manually.
        """
        self.set_edges(edges)
        if custom:
            return
        # TODO: Add functionality to set funcs automatically in the same way as
        # in the LocalFunctionSpace class

    def set_edges(self, edges: Union[MeshCell, list[Edge]]) -> None:
        """
        Set the edges on which the trace is defined.

        Parameters
        ----------
        edges : Union[MeshCell, list[Edge]]
            The edges on which the trace is defined.
        """
        if isinstance(edges, MeshCell):
            self.edges = edges.get_edges()
        elif isinstance(edges, list):
            if not all(isinstance(edge, Edge) for edge in edges):
                raise ValueError("All elements must be of type Edge")
            self.edges = edges
        else:
            raise ValueError("'edges' must be of type MeshCell or list[Edge]")
        self.num_edges = len(self.edges)

    def set_trace_values_on_edge(
        self, edge_index: int, values: np.ndarray
    ) -> None:
        """
        Set the values of the trace on a specific edge.

        Parameters
        ----------
        edge_index : int
            The index of the edge on which the values are set.
        values : np.ndarray
            The values of the trace on the edge.
        """
        if edge_index < 0 or edge_index >= self.num_edges:
            raise ValueError("The edge index is out of range")
        edge = self.edges[edge_index]
        if values.shape[0] != edge.num_pts:
            raise ValueError(
                "The number of values must match the number of points"
            )
        start, end = self.edge_sampled_indices[edge_index]
        self.values[start:end] = values

    def set_trace_values(self, values: Union[int, float, np.ndarray]) -> None:
        """
        Set the values of the trace on all edges.

        Parameters
        ----------
        values : Union[int, float, np.ndarray]
            The values of the trace on the edges.
        """
        if isinstance(values, (int, float)):
            self.values = np.array([values for _ in range(self.num_pts)])
        elif isinstance(values, np.ndarray):
            if values.shape[0] != self.num_pts:
                raise ValueError(
                    "The number of values must match the number of points"
                )
            self.values = values

    def set_funcs(self, funcs: Union[Func_R2_R, list[Func_R2_R]]) -> None:
        """
        Set the functions used to define the trace.

        Parameters
        ----------
        funcs : Union[Func_R2_R, list[Func_R2_R]]
            The functions used to define the trace.
        """
        if not isinstance(funcs, list):
            if is_Func_R2_R(funcs):
                self.funcs = [funcs for _ in range(self.num_edges)]
                return
        if isinstance(funcs, list):
            for func in funcs:
                if not is_Func_R2_R(func):
                    raise ValueError(
                        "All elements must be of callable maps from R^2 to R"
                    )
            self.funcs = funcs
        raise ValueError("'funcs' must be of type Func_R2_R or list[Func_R2_R]")

    def set_funcs_from_polys(
        self, polys: Union[Polynomial, list[Polynomial]]
    ) -> None:
        """
        Set the functions used to define the trace from a list of polynomials.

        Parameters
        ----------
        polys : Union[Polynomial, list[Polynomial]]
            The polynomials used to define the trace.
        """
        if isinstance(polys, Polynomial):
            self.set_funcs(polys.eval)  # type: ignore
        if not isinstance(polys, list):
            raise ValueError("'polys' must be of type list[Polynomial]")
        if not all(isinstance(poly, Polynomial) for poly in polys):
            raise ValueError("All elements must be of type Polynomial")
        self.set_funcs([poly.eval for poly in polys])  # type: ignore

    def edges_are_parametrized(self) -> bool:
        """
        Check if all edges are parametrized.

        Returns
        -------
        bool
            True if all edges are parametrized, False otherwise.
        """
        return all(edge.is_parameterized for edge in self.edges)

    def find_num_pts(self) -> None:
        """
        Find the number of points on the edges.

        Raises
        ------
        ValueError
            If the edges are not parametrized.
        """
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.num_pts = sum(edge.num_pts - 1 for edge in self.edges)

    def find_edge_sampled_indices(self) -> None:
        """
        Find the start and end indices of the edges in the values array.

        Raises
        ------
        ValueError
            If the edges are not parametrized.
        """
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.edge_sampled_indices = []
        start = 0
        for edge in self.edges:
            end = start + edge.num_pts - 1
            self.edge_sampled_indices.append((start, end))
            start = end

    def find_values(self) -> None:
        """
        Find the values of the trace on the edges by evaluating the trace
        functions at the sampled points on each edge.

        Raises
        ------
        ValueError
            If the functions are not set.
        ValueError
            If the edges are not parametrized.
        """
        if not hasattr(self, "funcs"):
            raise ValueError("The functions must be set before evaluating")
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.find_num_pts()
        self.find_edge_sampled_indices()
        self.values = np.zeros(self.num_pts)
        for k, edge in enumerate(self.edges):
            start, end = self.edge_sampled_indices[k]
            self.values[start:end] = edge.evaluate_function(
                self.funcs[k], ignore_endpoint=True
            )
