"""
test_edge_space.py
==================

Tests for the edge_space module.
"""

import puncturedfem as pf

MAX_DEG = 3


def get_edge_space_dim(p: int, m: int) -> int:
    """
    Returns the dimension of the edge space on a line with p + 1 vertices and m
    edges.
    """
    p_plus_2_choose_2 = ((p + 2) * (p + 1)) // 2
    if m < 1:
        return p_plus_2_choose_2
    p_plus_2_minus_m_choose_2 = ((p + 2 - m) * (p + 1 - m)) // 2
    return p_plus_2_choose_2 - p_plus_2_minus_m_choose_2


def check_dim(p: int, m: int, edge: pf.Edge) -> None:
    """
    Check that computed and predicted dimensions are equal.
    """
    dim = get_edge_space_dim(p, m)
    edge_space = pf.EdgeSpace(edge, p)
    print(
        f"{edge.curve_type} (p = {p}): "
        + f"predicted = {dim} / found = {edge_space.num_funs}"
    )
    if edge.is_loop:
        assert edge_space.num_vert_funs == 0
    else:
        assert edge_space.num_vert_funs == 2
    assert edge_space.num_funs == dim


def test_edge_space_dim_line() -> None:
    """
    Checks that the dimension of the edge space on a line is given by
    (p + 2 choose 2) - (p + 1 choose 2) = p + 1.
    """
    m = 1
    anchor = pf.Vert(1.0, 0.0)
    endpnt = pf.Vert(2.0, 1.0)
    edge = pf.Edge(anchor, endpnt, idx=0)
    quad_dict = pf.get_quad_dict(n=32)
    edge.parameterize(quad_dict)
    for p in range(1, MAX_DEG + 1):
        check_dim(p, m, edge)


def test_edge_space_dim_circular_arc() -> None:
    """
    Checks that the dimension of the edge space on a circular arc is given by
    (p + 2 choose 2) - (p choose 2) = 2p + 1.
    """
    m = 2
    anchor = pf.Vert(1.0, 0.0)
    endpnt = pf.Vert(2.0, 1.0)
    edge = pf.Edge(
        anchor, endpnt, idx=0, curve_type="circular_arc_deg", theta0=180
    )
    quad_dict = pf.get_quad_dict(n=32)
    edge.parameterize(quad_dict)
    for p in range(1, MAX_DEG + 1):
        check_dim(p, m, edge)


def test_edge_space_dim_ellipse() -> None:
    """
    Checks that the dimension of the edge space on a circle is given by
    (p + 2 choose 2) - (p choose 2) = 2p + 1.
    """
    m = 2
    anchor = pf.Vert(1.0, 0.0)
    edge = pf.Edge(anchor, anchor, idx=0, curve_type="ellipse", a=1.23, b=0.456)
    quad_dict = pf.get_quad_dict(n=32)
    edge.parameterize(quad_dict)
    for p in range(1, MAX_DEG + 1):
        check_dim(p, m, edge)


def test_edge_space_dim_sine_wave() -> None:
    """
    Checks that the dimension of the edge space on a sine wave is given by
    (p + 2 choose 2).
    """
    m = -1
    anchor = pf.Vert(1.0, 0.0)
    endpnt = pf.Vert(2.0, 1.0)
    edge = pf.Edge(
        anchor, endpnt, idx=0, curve_type="sine_wave", amp=1.23, freq=5
    )
    quad_dict = pf.get_quad_dict(n=32)
    edge.parameterize(quad_dict)
    for p in range(1, MAX_DEG + 1):
        check_dim(p, m, edge)
