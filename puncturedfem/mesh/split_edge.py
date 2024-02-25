"""
split_edge.py
=============

Module containing the split_edge() function.
"""

import numpy as np

from .edge import Edge
from .vert import Vert
from . import transform


def split_edge(e: Edge, t_split: float = np.pi) -> tuple[Edge, Edge]:
    """
    Splits the edge e and returns the two resulting edges.
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
