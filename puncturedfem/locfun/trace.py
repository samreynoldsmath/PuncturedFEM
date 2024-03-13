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

    Usage
    -----
    TODO: Add usage examples
    """

    edges: list[Edge]
    num_pts: int
    values: np.ndarray
    funcs: list[Func_R2_R]

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
        else:
            raise ValueError("'edges' must be of type MeshCell or list[Edge]")

    def set_trace_values(self, values: Union[int, float, np.ndarray]) -> None:
        if isinstance(values, (int, float)):
            self.values = np.array([values for _ in range(self.num_pts)])
        elif isinstance(values, np.ndarray):
            if values.shape[0] != self.num_pts:
                raise ValueError(
                    "The number of values must match the number of points"
                )
            self.values = values

    def set_funcs(self, funcs: list[Func_R2_R]) -> None:
        for func in funcs:
            if not DirichletTrace.check_func(func):
                raise ValueError(
                    "All elements must be of callable maps from R^2 to R"
                )
        self.funcs = funcs

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

    def set_funcs_from_polys(self, polys: list[Polynomial]) -> None:
        if not isinstance(polys, list):
            raise ValueError("'polys' must be of type list[Polynomial]")
        if not all(isinstance(poly, Polynomial) for poly in polys):
            raise ValueError("All elements must be of type Polynomial")
        self.funcs = [poly.eval for poly in polys]  # type: ignore

    def edges_are_parametrized(self) -> bool:
        return all(edge.is_parameterized for edge in self.edges)

    def find_num_pts(self) -> None:
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.num_pts = sum(edge.num_pts for edge in self.edges)

    def find_values(self) -> None:
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.values = np.zeros(self.num_pts)
        start = 0
        for k, edge in enumerate(self.edges):
            end = start + edge.num_pts
            self.values[start:end] = edge.evaluate_function(self.funcs[k])
            start = end
