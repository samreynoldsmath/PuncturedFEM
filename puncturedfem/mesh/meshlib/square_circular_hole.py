"""
Planar mesh with a square domain and a circular hole.

Routines in this module
-----------------------
square_circular_hole _get_verts _get_edges

Notes
-----
- The square is defined by the vertices (0, 0), (1, 0), (1, 1), and (0, 1).
- The circular hole is defined by the center (0.5, 0.5) and radius 0.25.
- The mesh consists of 2 cells. The square domain is cell 0 and the circular
  hole is cell 1.
"""

from ..edge import Edge
from ..planar_mesh import PlanarMesh
from ..vert import Vert
from .__builder__ import mesh_builder


def square_circular_hole(verbose: bool = True) -> PlanarMesh:
    """
    Get a mesh consisting of a square with a circular hole.

    Parameters
    ----------
    verbose : bool, optional
        If True, print mesh information. Default is True.

    Returns
    -------
    PlanarMesh
        A planar mesh consisting of a square with a circular hole.

    Notes
    -----
    - The mesh consists of 2 cells.
    """
    return mesh_builder(_get_verts, _get_edges, verbose=verbose)


def _get_verts() -> list[Vert]:
    verts: list[Vert] = []
    verts.append(Vert(x=0.0, y=0.0))
    verts.append(Vert(x=1.0, y=0.0))
    verts.append(Vert(x=1.0, y=1.0))
    verts.append(Vert(x=0.0, y=1.0))
    verts.append(Vert(x=0.5, y=0.5))  # center of circle
    return verts


def _get_edges(verts: list[Vert]) -> list[Edge]:
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
