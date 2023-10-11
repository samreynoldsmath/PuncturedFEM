"""
build_quad.py
=============

Build quadrature rules for testing purposes.
"""


from puncturedfem import Quad


def get_quad_dict(n: int = 64):
    """
    Set up test parameters
    """
    q_trap = Quad(qtype="trap", n=n)
    q_kress = Quad(qtype="kress", n=n)
    quad_dict = {"kress": q_kress, "trap": q_trap}
    return quad_dict
