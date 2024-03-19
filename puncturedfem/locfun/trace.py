"""
trace.py
========

This module contains the class `DirichletTrace` which is used to represent the
trace of a LocalFunction on the boundary of a MeshCell.
"""

from typing import Optional, Union

import numpy as np

from ..mesh.cell import MeshCell
from ..mesh.edge import Edge
from ..util.types import FloatLike, Func_R2_R, is_Func_R2_R
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
    See examples/ex0b-trace.ipynb for a tutorial on how to use this class.
    """

    edges: list[Edge]
    num_edges: int
    num_pts: int
    values: np.ndarray
    funcs: list[Func_R2_R]
    edge_sampled_indices: list[tuple[int, int]]

    def __init__(
        self,
        edges: Union[MeshCell, list[Edge]],
        custom: bool = False,
        funcs: Optional[Union[Func_R2_R, list[Func_R2_R]]] = None,
        values: Optional[Union[FloatLike, list[FloatLike]]] = None,
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
        funcs : Optional[Union[Func_R2_R, list[Func_R2_R]]], optional
            The functions used to define the trace. Default is None.
        values : Optional[Union[FloatLike, list[FloatLike]]], optional
            The values of the trace on the edges. Default is None. Takes
            precedence over funcs.
        """
        self.set_edges(edges)
        if self.edges_are_parametrized():
            self.find_num_pts()
            self.find_edge_sampled_indices()
        if custom:
            return
        if values is not None:
            self.set_trace_values(values)
            return
        if funcs is not None:
            self.set_funcs(funcs)
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

    def get_edge_sampled_point_indices(
        self, edge_index: int
    ) -> tuple[int, int]:
        """Return the indices of the sampled points on given edge"""
        return self.edge_sampled_indices[edge_index]

    def set_trace_values_on_edge(
        self, edge_index: int, values: FloatLike
    ) -> None:
        """
        Set the values of the trace on a specific edge.

        Parameters
        ----------
        edge_index : int
            The index of the edge on which the values are set.
        values : FloatLike
            The values of the trace on the edge.
        """
        if edge_index < 0 or edge_index >= self.num_edges:
            raise ValueError("The edge index is out of range")
        edge = self.edges[edge_index]
        if isinstance(values, (int, float)):
            start, end = self.edge_sampled_indices[edge_index]
            self.values[start:end] = values
            return
        if not isinstance(values, np.ndarray):
            raise ValueError(
                "'values' must be of type int, float, or np.ndarray"
            )
        if values.shape[0] != edge.num_pts - 1:
            raise ValueError(
                "The number of values must match the number of points"
            )
        start, end = self.edge_sampled_indices[edge_index]
        self.values[start:end] = values

    def set_trace_values(
        self, values: Union[FloatLike, list[FloatLike]]
    ) -> None:
        """
        Set the values of the trace on all edges.

        Parameters
        ----------
        values : FloatLike
            The values of the trace on the edges. If a scalar, the same value
            is set on all edges. If a list, the length must be equal to the
            number of edges. If an np.ndarray, the shape must be equal to the
            number of points on the edges.
        """
        if isinstance(values, (int, float)):
            values = np.zeros(self.num_pts) + values
        if isinstance(values, np.ndarray):
            if values.shape[0] != self.num_pts:
                raise ValueError(
                    "The number of values must match the number of points"
                )
            self.values = values
            return
        if not isinstance(values, list):
            raise ValueError(
                "'values' must be of type int, float, or np.ndarray or a list "
                "of values"
            )
        if len(values) != self.num_edges:
            raise ValueError(
                "'values' must be a scalar or a list of values of length "
                "equal to the number of edges"
            )
        self.values = np.zeros(self.num_pts)
        for k in range(self.num_edges):
            self.set_trace_values_on_edge(k, values[k])

    def set_func_on_edge(self, edge_index: int, func: Func_R2_R) -> None:
        """
        Set the function used to define the trace on a specific edge.

        Parameters
        ----------
        edge_index : int
            The index of the edge on which the function is set.
        func : Func_R2_R
            The function used to define the trace on the edge.
        """
        if not isinstance(edge_index, int):
            raise ValueError("'edge_index' must be of type int")
        if edge_index < 0 or edge_index >= self.num_edges:
            raise ValueError("The edge index is out of range")
        if not is_Func_R2_R(func):
            raise ValueError("The function must be a map from R^2 to R")
        self.funcs[edge_index] = func

    def set_funcs(
        self,
        funcs: Union[Func_R2_R, list[Func_R2_R]],
        compute_vals: bool = True,
    ) -> None:
        """
        Set the functions used to define the trace.

        Parameters
        ----------
        funcs : Union[Func_R2_R, list[Func_R2_R]]
            The functions used to define the trace.
        """
        if not isinstance(funcs, list):
            # TODO: a Polynomial should be recognized as a callable map
            if isinstance(funcs, Polynomial):
                funcs = [funcs for _ in range(self.num_edges)]
            elif is_Func_R2_R(funcs):
                funcs = [funcs for _ in range(self.num_edges)]
        if isinstance(funcs, list):
            # TODO: a Polynomial should be recognized as a callable map
            is_accepted = True
            for func in funcs:
                if isinstance(func, Polynomial):
                    continue
                if not is_Func_R2_R(func):
                    is_accepted = False
                    break
            if not is_accepted:
                raise ValueError(
                    "All elements must be callable maps from R^2 to R"
                )
            if len(funcs) != self.num_edges:
                raise ValueError(
                    "The number of functions must match the number of edges"
                )
            self.funcs = funcs
            if compute_vals and self.edges_are_parametrized():
                self.find_values()
            return
        raise ValueError("'funcs' must be callable maps from R^2 to R")

    def set_func_from_poly_on_edge(
        self, edge_index: int, poly: Polynomial, compute_vals: bool = True
    ) -> None:
        """
        Set the function used to define the trace on a specific edge from a
        polynomial.

        Parameters
        ----------
        edge_index : int
            The index of the edge on which the function is set.
        poly : Polynomial
            The polynomial used to define the trace on the edge.
        """
        if not isinstance(edge_index, int):
            raise ValueError("'edge_index' must be of type int")
        if edge_index < 0 or edge_index >= self.num_edges:
            raise ValueError("The edge index is out of range")
        if not isinstance(poly, Polynomial):
            raise ValueError("'poly' must be of type Polynomial")
        self.funcs[edge_index] = poly.eval  # type: ignore
        if compute_vals and self.edges_are_parametrized():
            self.find_values()

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
        if not hasattr(self, "edges"):
            return False
        return all(edge.is_parameterized for edge in self.edges)

    def find_num_pts(self) -> None:
        """
        Find the number of points on the edges.
        """
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.num_pts = sum(edge.num_pts - 1 for edge in self.edges)

    def find_edge_sampled_indices(self) -> None:
        """
        Find the start and end indices of the edges in the values array.
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
            self.values[start:end] = edge.evaluate_function(self.funcs[k])
