"""
Dirichlet trace.

Classes
-------
DirichletTrace
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..mesh.cell import MeshCell
from ..mesh.edge import Edge
from ..util.types import FloatLike, Func_R2_R, is_Func_R2_R
from .poly.poly import Polynomial


class DirichletTrace:
    """
    Dirichlet trace.

    This class is used to represent the trace of a LocalFunction on the boundary
    of a MeshCell. The trace is represented as a list of edges, and the values
    of the trace on the edges are stored in an array. The class also contains
    methods to set the values of the trace and to evaluate the trace on the
    edges. The weighted normal and tangential derivatives can also be set, but
    must be computed separately.

    Attributes
    ----------
    edges : list[Edge]
        The edges on which the trace is defined.
    num_pts : int
        The number of points on the edges.
    values : np.ndarray
        The values of the trace on the edges.
    w_norm_deriv : Optional[np.ndarray]
        The weighted normal derivatives of the trace on the edges.
    w_tang_deriv : Optional[np.ndarray]
        The weighted tangential derivatives of the trace on the edges.
    funcs : list[Func_R2_R]
        The functions used to define the trace.
    edge_sampled_indices : list[tuple[int, int]]
        The start and end indices of the edges in the values array.
    """

    edges: list[Edge]
    num_edges: int
    num_pts: int
    values: np.ndarray
    w_norm_deriv: Optional[np.ndarray]
    w_tang_deriv: Optional[np.ndarray]
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
        Construct a DirichletTrace object.

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
        self.w_norm_deriv = None
        self.w_tang_deriv = None
        if custom:
            return
        if values is not None:
            self.set_trace_values(values)
            return
        if funcs is not None:
            self.set_funcs(funcs)
        else:
            self.funcs = [lambda x, y: 0 for _ in range(self.num_edges)]

    def __add__(self, other: DirichletTrace) -> DirichletTrace:
        """
        Add two Dirichlet traces.

        Parameters
        ----------
        other : DirichletTrace
            The other Dirichlet trace.

        Returns
        -------
        DirichletTrace
            The sum of the two Dirichlet traces.
        """
        if not isinstance(other, DirichletTrace):
            raise ValueError("The other trace must be a DirichletTrace")
        if self.num_edges != other.num_edges:
            raise ValueError("The number of edges must be the same")
        new = DirichletTrace(self.edges, custom=True)
        new.set_trace_values(self.values + other.values)
        if self.w_norm_deriv is not None and other.w_norm_deriv is not None:
            new.set_weighted_normal_derivative(
                self.w_norm_deriv + other.w_norm_deriv
            )
        if self.w_tang_deriv is not None and other.w_tang_deriv is not None:
            new.set_weighted_tangential_derivative(
                self.w_tang_deriv + other.w_tang_deriv
            )
        return new

    def __mul__(self, other: Union[int, float]) -> DirichletTrace:
        """
        Multiply the trace by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar to multiply the trace by.

        Returns
        -------
        DirichletTrace
            The trace multiplied by the scalar.
        """
        if not isinstance(other, (int, float)):
            raise ValueError("The scalar must be an int or float")
        new = DirichletTrace(self.edges, custom=True)
        new.set_trace_values(self.values * other)
        if self.w_norm_deriv is not None:
            new.set_weighted_normal_derivative(self.w_norm_deriv * other)
        if self.w_tang_deriv is not None:
            new.set_weighted_tangential_derivative(self.w_tang_deriv * other)
        return new

    def __rmul__(self, other: Union[int, float]) -> DirichletTrace:
        """
        Multiply the trace by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar to multiply the trace by.

        Returns
        -------
        DirichletTrace
            The trace multiplied by the scalar.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float]) -> DirichletTrace:
        """
        Divide the trace by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar to divide the trace by.

        Returns
        -------
        DirichletTrace
            The trace divided by the scalar.
        """
        if not isinstance(other, (int, float)):
            raise ValueError("The scalar must be an int or float")
        if other == 0:
            raise ValueError("The scalar must be non-zero")
        new = DirichletTrace(self.edges, custom=True)
        new.set_trace_values(self.values / other)
        if self.w_norm_deriv is not None:
            new.set_weighted_normal_derivative(self.w_norm_deriv / other)
        if self.w_tang_deriv is not None:
            new.set_weighted_tangential_derivative(self.w_tang_deriv / other)
        return new

    def __sub__(self, other: DirichletTrace) -> DirichletTrace:
        """
        Subtract two Dirichlet traces.

        Parameters
        ----------
        other : DirichletTrace
            The other Dirichlet trace.

        Returns
        -------
        DirichletTrace
            The difference of the two Dirichlet traces.
        """
        return self.__add__(other.__mul__(-1))

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
        """
        Get the indices of the sampled points on given edge.

        Parameters
        ----------
        edge_index : int
            The index of the edge.

        Returns
        -------
        tuple[int, int]
            The start and end indices of the sampled points on the edge.
        """
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
        num_edge_pts = self.edges[edge_index].num_pts
        self._validate_values_on_edge(values, num_edge_pts - 1)
        start, end = self.edge_sampled_indices[edge_index]
        self.values[start:end] = values

    def _validate_values_on_edge(self, values: FloatLike, num_pts: int) -> None:
        if isinstance(values, (int, float)):
            return
        if not isinstance(values, np.ndarray):
            raise ValueError("'values' to be set on edge must be FloatLike")
        if values.shape[0] != num_pts:
            raise ValueError(
                "The number of values must match the number of points on edge"
            )

    def set_trace_values(
        self, values: Union[FloatLike, list[FloatLike]]
    ) -> None:
        """
        Set the values of the trace on all edges.

        Parameters
        ----------
        values : FloatLike or list[FloatLike]
            The values of the trace on the edges. If a scalar, the same value
            is set on all edges. If a list, the length must be equal to the
            number of edges. If an np.ndarray, the shape must be equal to the
            number of points on the edges.
        """
        values = self._convert_values_to_list_floatlike(values)
        self.values = np.zeros(self.num_pts)
        for k in range(self.num_edges):
            self.set_trace_values_on_edge(k, values[k])

    def _convert_values_to_list_floatlike(
        self, values: Union[FloatLike, list[FloatLike]]
    ) -> list[FloatLike]:
        if isinstance(values, list):
            for item in values:
                if not isinstance(item, np.ndarray):
                    raise ValueError("item in list is not a numpy.ndarray")
            return values
        if isinstance(values, (int, float)):
            return [values for _ in range(self.num_edges)]
        if isinstance(values, np.ndarray):
            if values.shape[0] != self.num_pts:
                raise ValueError(
                    "The number of values must match the number of points"
                )
            return [
                values[start:end] for start, end in self.edge_sampled_indices
            ]
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
        raise ValueError("values must be FloatLike or list[FloatLike]")

    def set_weighted_normal_derivative_on_edge(
        self, edge_index: int, wnd: FloatLike
    ) -> None:
        """
        Set the weighted normal derivative of the trace on a specific edge.

        Parameters
        ----------
        edge_index : int
            The index of the edge on which the weighted normal derivative is
            set.
        wnd : FloatLike
            The weighted normal derivative of the trace on the edge.
        """
        if edge_index < 0 or edge_index >= self.num_edges:
            raise ValueError("The edge index is out of range")
        num_edge_pts = self.edges[edge_index].num_pts
        self._validate_values_on_edge(wnd, num_edge_pts - 1)
        start, end = self.edge_sampled_indices[edge_index]
        if self.w_norm_deriv is None:
            self.w_norm_deriv = np.zeros(self.num_pts)
        self.w_norm_deriv[start:end] = wnd

    def set_weighted_normal_derivative(
        self, wnd: Union[FloatLike, list[FloatLike]]
    ) -> None:
        """
        Set the weighted normal derivatives of the trace on all edges.

        Parameters
        ----------
        wnd : FloatLike or list[FloatLike]
            The weighted normal derivatives of the trace on the edges. If a
            scalar, the same value is set on all edges. If a list, the length
            must be equal to the number of edges. If an np.ndarray, the shape
            must be equal to the number of points on the edges.
        """
        wnd = self._convert_values_to_list_floatlike(wnd)
        self.w_norm_deriv = np.zeros(self.num_pts)
        for k in range(self.num_edges):
            self.set_weighted_normal_derivative_on_edge(k, wnd[k])

    def set_weighted_tangential_derivative_on_edge(
        self, edge_index: int, wtd: FloatLike
    ) -> None:
        """
        Set the weighted tangential derivative of the trace on a specific edge.

        Parameters
        ----------
        edge_index : int
            The index of the edge on which the weighted tangential derivative is
            set.
        wtd : FloatLike
            The weighted tangential derivative of the trace on the edge.
        """
        if edge_index < 0 or edge_index >= self.num_edges:
            raise ValueError("The edge index is out of range")
        num_edge_pts = self.edges[edge_index].num_pts
        self._validate_values_on_edge(wtd, num_edge_pts - 1)
        start, end = self.edge_sampled_indices[edge_index]
        if self.w_tang_deriv is None:
            self.w_tang_deriv = np.zeros(self.num_pts)
        self.w_tang_deriv[start:end] = wtd

    def set_weighted_tangential_derivative(
        self, wtd: Union[FloatLike, list[FloatLike]]
    ) -> None:
        """
        Set the weighted tangential derivatives of the trace on all edges.

        Parameters
        ----------
        wtd : FloatLike or list[FloatLike]
            The weighted tangential derivatives of the trace on the edges. If a
            scalar, the same value is set on all edges. If a list, the length
            must be equal to the number of edges. If an np.ndarray, the shape
            must be equal to the number of points on the edges.
        """
        wtd = self._convert_values_to_list_floatlike(wtd)
        self.w_tang_deriv = np.zeros(self.num_pts)
        for k in range(self.num_edges):
            self.set_weighted_tangential_derivative_on_edge(k, wtd[k])

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
        # if not is_Func_R2_R(func):
        #     raise ValueError("The function must be a map from R^2 to R")
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
        compute_vals : bool, optional
            A flag indicating whether the values of the trace should be
            computed. Default is True.
        """
        if not isinstance(funcs, list):
            if isinstance(funcs, Polynomial):
                funcs = [funcs for _ in range(self.num_edges)]
            elif is_Func_R2_R(funcs):
                funcs = [funcs for _ in range(self.num_edges)]
        if isinstance(funcs, list):
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
        Set the polynomial used to define the trace on a specific edge.

        Parameters
        ----------
        edge_index : int
            The index of the edge on which the function is set.
        poly : Polynomial
            The polynomial used to define the trace on the edge.
        compute_vals : bool, optional
            A flag indicating whether the values of the trace should be
            computed. Default is True.
        """
        if not isinstance(edge_index, int):
            raise ValueError("'edge_index' must be of type int")
        if edge_index < 0 or edge_index >= self.num_edges:
            raise ValueError("The edge index is out of range")
        if not isinstance(poly, Polynomial):
            raise ValueError("'poly' must be of type Polynomial")
        self.funcs[edge_index] = poly  # type: ignore
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
        Find the total number of sampled points on the edges.

        The result is stored in the attribute 'num_pts'.
        """
        if not self.edges_are_parametrized():
            raise ValueError("All edges must be parametrized")
        self.num_pts = sum(edge.num_pts - 1 for edge in self.edges)

    def find_edge_sampled_indices(self) -> None:
        """
        Find the start and end indices of the edges in the values array.

        The result is stored in the attribute 'edge_sampled_indices'.
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
        Find the values of the trace on the edges.

        This is done by evaluating the trace functions at the sampled points on
        each edge.

        The result is stored in the attribute 'values'.
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
