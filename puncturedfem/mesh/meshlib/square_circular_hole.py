"""
square_circular_hole.py
=======================

Module containing the square_circular_hole function, which returns a planar mesh
with a square domain and a circular hole.
"""

from ..edge import Edge
from ..planar_mesh import PlanarMesh
from ..vert import Vert
from .__builder__ import mesh_builder


def square_circular_hole(verbose: bool = True) -> PlanarMesh:
    """Returns the square_circular_hole mesh."""
    return mesh_builder(get_verts, get_edges, verbose=verbose)


# VERTICES ###################################################################


def get_verts() -> list[Vert]:
    """Returns a list of Vertices for the mesh."""
    Verts: list[Vert] = []
    Verts.append(Vert(x=0.0, y=0.0))
    Verts.append(Vert(x=2.0, y=0.0))
    Verts.append(Vert(x=2.0, y=1.0))
    Verts.append(Vert(x=0.0, y=1.0))
    Verts.append(Vert(x=0.5, y=0.5))  # center of circle
    return Verts


def get_edges(Verts: list[Vert]) -> list[Edge]:
    """Returns a list of edges for the mesh."""
    edges: list[Edge] = []
    edges.append(Edge(Verts[0], Verts[1], pos_cell_idx=0))
    edges.append(Edge(Verts[1], Verts[2], pos_cell_idx=0))
    edges.append(Edge(Verts[2], Verts[3], pos_cell_idx=0))
    edges.append(Edge(Verts[3], Verts[0], pos_cell_idx=0))
    edges.append(
        Edge(
            Verts[4],
            Verts[4],
            pos_cell_idx=1,
            neg_cell_idx=0,
            curve_type="circle",
            quad_type="trap",
            radius=0.25,
        )
    )
    return edges
