"""
trace.py
========

This module contains the class `DirichletTrace` which is used to represent the
trace of a LocalFunction on the boundary of a MeshCell.

TODO: Deprecate the PiecewisePolynomial class
"""

import inspect
from typing import Union, Callable

import numpy as np

from ..mesh.edge import Edge
from ..mesh.cell import MeshCell
from .poly.poly import Polynomial

FloatLike = Union[int, float, np.ndarray]
Func_R2_R = Callable[[FloatLike, FloatLike], FloatLike]


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
        # TODO: add method docstrings
        self.set_edges(edges)
        if custom:
            return
        # TODO: Add functionality to set funcs automatically in the same way as
        # in the LocalFunctionSpace class

    def set_edges(self, edges: Union[MeshCell, list[Edge]]) -> None:
        if isinstance(edges, MeshCell):
            self.edges = edges.get_edges()
        elif isinstance(edges, list):
            if not all(isinstance(edge, Edge) for edge in edges):
                raise ValueError("All elements must be of type Edge")
            self.edges = edges
            self.num_edges = len(edges)
        else:
            raise ValueError("'edges' must be of type MeshCell or list[Edge]")

    def set_trace_values_on_edge(
        self, edge_index: int, values: np.ndarray
    ) -> None:
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
        if isinstance(values, (int, float)):
            self.values = np.array([values for _ in range(self.num_pts)])
        elif isinstance(values, np.ndarray):
            if values.shape[0] != self.num_pts:
                raise ValueError(
                    "The number of values must match the number of points"
                )
            self.values = values

    def set_funcs(self, funcs: Union[Func_R2_R, list[Func_R2_R]]) -> None:
        if not isinstance(funcs, list):
            if DirichletTrace.check_func(funcs):
                self.funcs = [funcs for _ in range(self.num_edges)]
                return
        if isinstance(funcs, list):
            for func in funcs:
                if not DirichletTrace.check_func(func):
                    raise ValueError(
                        "All elements must be of callable maps from R^2 to R"
                    )
            self.funcs = funcs
        raise ValueError("'funcs' must be of type Func_R2_R or list[Func_R2_R]")

    @staticmethod
    def check_func(func: Func_R2_R) -> bool:
        if not callable(func):
            return False
        sig = inspect.signature(func)
        params = sig.parameters.values()
        return len(params) == 2 and all(
            isinstance(param.annotation, type)
            and issubclass(param.annotation, (int, float, np.ndarray))
            for param in params
        )

    def set_funcs_from_polys(
        self, polys: Union[Polynomial, list[Polynomial]]
    ) -> None:
        if isinstance(polys, Polynomial):
            self.set_funcs(polys.eval)  # type: ignore
        if not isinstance(polys, list):
            raise ValueError("'polys' must be of type list[Polynomial]")
        if not all(isinstance(poly, Polynomial) for poly in polys):
            raise ValueError("All elements must be of type Polynomial")
        self.set_funcs([poly.eval for poly in polys])  # type: ignore

    def edges_are_parametrized(self) -> bool:
        return all(edge.is_parameterized for edge in self.edges)

    def find_num_pts(self) -> None:
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.num_pts = sum(edge.num_pts for edge in self.edges)

    def find_edge_sampled_indices(self) -> None:
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.edge_sampled_indices = []
        start = 0
        for edge in self.edges:
            end = start + edge.num_pts
            self.edge_sampled_indices.append((start, end))
            start = end

    def find_values(self) -> None:
        if not hasattr(self, "funcs"):
            raise ValueError("The functions must be set before evaluating")
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.find_num_pts()
        self.find_edge_indices()
        self.values = np.zeros(self.num_pts)
        for k, edge in enumerate(self.edges):
            start, end = self.edge_sampled_indices[k]
            self.values[start:end] = edge.evaluate_function(self.funcs[k])
