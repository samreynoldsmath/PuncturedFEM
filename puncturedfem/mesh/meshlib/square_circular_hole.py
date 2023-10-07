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
    verts: list[Vert] = []
    verts.append(Vert(x=0.0, y=0.0))
    verts.append(Vert(x=2.0, y=0.0))
    verts.append(Vert(x=2.0, y=1.0))
    verts.append(Vert(x=0.0, y=1.0))
    verts.append(Vert(x=0.5, y=0.5))  # center of circle
    return verts


def get_edges(verts: list[Vert]) -> list[Edge]:
    """Returns a list of edges for the mesh."""
    edges: list[Edge] = []
    edges.append(Edge(verts[0], verts[1], pos_cell_idx=0))
    edges.append(Edge(verts[1], verts[2], pos_cell_idx=0))
    edges.append(Edge(verts[2], verts[3], pos_cell_idx=0))
    edges.append(Edge(verts[3], verts[0], pos_cell_idx=0))
    edges.append(
        Edge(
            verts[4],
            verts[4],
            pos_cell_idx=1,
            neg_cell_idx=0,
            curve_type="circle",
            quad_type="trap",
            radius=0.25,
        )
    )
    return edges
