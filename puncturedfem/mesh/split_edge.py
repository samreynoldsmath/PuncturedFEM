"""
split_edge.py
=============

Module containing the split_edge() function.
"""

from typing import Optional, Union

import numpy as np

from . import transform
from .edge import Edge
from .vert import Vert


def split_edge(
    e: Edge,
    t_split: Optional[Union[int, float, list[int | float], np.ndarray]] = None,
    num_edges: Optional[int] = None,
) -> list[Edge] | tuple[Edge, Edge]:
    """
    Splits the edge e at the parameter t_split and returns the resulting edges.
    If t_split is a list of parameters, the edge is split at each parameter in
    the list. If num_edges is provided, the edge is split into num_edges equal
    parts.

    Parameters
    ----------
    e : Edge
        The edge to split.
    t_split : float, list of floats, or np.ndarray, optional
        The parameter(s) at which to split the edge. If a list of floats is
        provided, the edge is split at each parameter in the list. If a float is
        provided, the edge is split at that parameter. If None, num_edges must
        be provided. The default is None.
    num_edges : int, optional
        The number of equal parts to split the edge into. If None, t_split must
        be provided. The default is None.

    Returns
    -------
    list[Edge] or tuple[Edge, Edge]
        If t_split is a list of floats, the resulting edges are returned in a
        list. If t_split is a float, the resulting edges are returned as a
        tuple. If num_edges is provided, the resulting edges are returned in a
        list.

    Raises
    ------
    ValueError
        If t_split and num_edges are both None.
        If num_edges is less than 2.
        If t_split is not a float, list of floats, or np.ndarray.
    """
    if t_split is None and num_edges is None:
        raise ValueError("Either t_split or num_edges must be provided")
    if isinstance(num_edges, int):
        if num_edges < 2:
            raise ValueError("num_edges must be at least 2")
        t_split = np.linspace(e.t_bounds[0], e.t_bounds[1], num_edges + 1)[1:-1]
    if isinstance(t_split, (int, float)):
        return _split_edge_single(e, t_split)
    if isinstance(t_split, np.ndarray):
        t_split = t_split.tolist()
    if isinstance(t_split, list):
        if all(isinstance(t, float) for t in t_split):
            return _split_edge_multiple(e, t_split)
    raise ValueError("t_split must be a float or a list of floats")


def _split_edge_multiple(e: Edge, t_split: list[float]) -> list[Edge]:
    """
    Splits the edge e at the parameters in t_split and returns the resulting
    edges
    """
    t_split = list(set(t_split))
    t_split.sort()

    edges = []
    for t in t_split:
        e1, e2 = _split_edge_single(e, t)
        edges.append(e1)
        e = e2
    edges.append(e2)
    return edges


def _split_edge_single(e: Edge, t_split: float = np.pi) -> tuple[Edge, Edge]:
    """
    Splits the edge e at the parameter t_split and returns the two resulting
    edges
    """
    a, b = e.t_bounds
    gamma = e.get_parameterization_module()
    T_split = np.array([a, t_split, b])
    vert_coords = gamma.X(T_split, **e.curve_opts)

    for method_name, args in e.diary:
        method = getattr(transform, method_name)
        vert_coords = method(vert_coords, *args)

    new_anchor = Vert(x=vert_coords[0, 0], y=vert_coords[1, 0])
    split_vert = Vert(x=vert_coords[0, 1], y=vert_coords[1, 1])
    new_endpnt = Vert(x=vert_coords[0, 2], y=vert_coords[1, 2])

    e1 = Edge(
        anchor=new_anchor,
        endpnt=split_vert,
        pos_cell_idx=e.pos_cell_idx,
        neg_cell_idx=e.neg_cell_idx,
        curve_type=e.curve_type,
        quad_type="kress",
        t_bounds=(a, t_split),
        **e.curve_opts
    )

    e2 = Edge(
        anchor=split_vert,
        endpnt=new_endpnt,
        pos_cell_idx=e.pos_cell_idx,
        neg_cell_idx=e.neg_cell_idx,
        curve_type=e.curve_type,
        quad_type="kress",
        t_bounds=(t_split, b),
        **e.curve_opts
    )

    return e1, e2
