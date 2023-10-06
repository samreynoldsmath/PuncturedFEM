"""
pacman_subdiv.py
================

Module containing the pacman_subdiv function, which returns a planar mesh
representing a subdivision of the pacman mesh. The vertical edges have been
split into two edges each.
"""

from numpy import sqrt

from ..edge import Edge
from ..planar_mesh import PlanarMesh
from ..vert import Vert
from .__builder__ import mesh_builder

ROOT3OVER2 = sqrt(3) / 2
PACMAN_SCALE = 0.4
PACMAN_XSHIFT = 0.5
PACMAN_YSHIFT = 0.5
GHOST_SCALE = 0.6
GHOST_X_SHIFT = 3.5
GHOST_Y_SHIFT = 0.5


def pacman_subdiv(verbose: bool = True) -> PlanarMesh:
    """
    Returns a planar mesh representing a subdivision of the pacman mesh. The
    vertical edges have been split into two edges each.
    """
    return mesh_builder(get_verts, get_edges, verbose=verbose)


# VERTICES ###################################################################


def get_verts() -> list[Vert]:
    """Returns a list of Vertices for the mesh."""

    # define Vertices
    Verts: list[Vert] = []

    # rectangle corners
    Verts.append(Vert(x=0.0, y=0.0))  # 0
    Verts.append(Vert(x=1.0, y=0.0))  # 1
    Verts.append(Vert(x=3.0, y=0.0))  # 2
    Verts.append(Vert(x=4.0, y=0.0))  # 3
    Verts.append(Vert(x=4.0, y=1.0))  # 4
    Verts.append(Vert(x=3.0, y=1.0))  # 5
    Verts.append(Vert(x=1.0, y=1.0))  # 6
    Verts.append(Vert(x=0.0, y=1.0))  # 7

    # "Pac-Man"
    Verts.append(Vert(x=0.5, y=0.5))  # 8
    Verts.append(
        Vert(
            x=PACMAN_XSHIFT + PACMAN_SCALE * ROOT3OVER2,
            y=PACMAN_YSHIFT + PACMAN_SCALE * 0.5,
        )
    )  # 9
    Verts.append(
        Vert(
            x=PACMAN_XSHIFT + PACMAN_SCALE * ROOT3OVER2,
            y=PACMAN_YSHIFT - PACMAN_SCALE * 0.5,
        )
    )  # 10
    Verts.append(
        Vert(
            x=PACMAN_XSHIFT + PACMAN_SCALE * -0.1,
            y=PACMAN_YSHIFT + PACMAN_SCALE * 0.5,
        )
    )  # 11

    # central "dots"
    Verts.append(Vert(x=1.5, y=0.5))  # 12
    Verts.append(Vert(x=2.0, y=0.5))  # 13
    Verts.append(Vert(x=2.5, y=0.5))  # 14

    # "ghost"
    Verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (-0.5),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (-0.6),
        )
    )  # 15
    Verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (0.5),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (-0.6),
        )
    )  # 16
    Verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (0.5),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (0.2),
        )
    )  # 17
    Verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (-0.5),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (0.2),
        )
    )  # 18
    Verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (-0.25),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (0.1),
        )
    )  # 19
    Verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (0.25),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (0.1),
        )
    )  # 20

    # split vertical edges
    Verts.append(Vert(x=1.0, y=0.5))  # 21
    Verts.append(Vert(x=3.0, y=0.5))  # 22

    return Verts


# EDGES ######################################################################


def get_edges(Verts: list[Vert]) -> list[Edge]:
    """Returns a list of edges for the mesh."""

    # define edges
    edges = []

    # rectangles
    edges.append(Edge(Verts[0], Verts[1], pos_cell_idx=0))
    edges.append(Edge(Verts[1], Verts[2], pos_cell_idx=3))
    edges.append(Edge(Verts[2], Verts[3], pos_cell_idx=7))
    edges.append(Edge(Verts[3], Verts[4], pos_cell_idx=7))
    edges.append(Edge(Verts[4], Verts[5], pos_cell_idx=7))
    edges.append(Edge(Verts[5], Verts[6], pos_cell_idx=3))
    edges.append(Edge(Verts[6], Verts[7], pos_cell_idx=0))
    edges.append(Edge(Verts[7], Verts[0], pos_cell_idx=0))
    edges.append(Edge(Verts[1], Verts[21], pos_cell_idx=0, neg_cell_idx=3))
    edges.append(Edge(Verts[21], Verts[6], pos_cell_idx=0, neg_cell_idx=3))
    edges.append(Edge(Verts[2], Verts[22], pos_cell_idx=3, neg_cell_idx=7))
    edges.append(Edge(Verts[22], Verts[5], pos_cell_idx=3, neg_cell_idx=7))

    # pacman
    edges.append(Edge(Verts[8], Verts[9], pos_cell_idx=1, neg_cell_idx=0))
    edges.append(
        Edge(
            Verts[9],
            Verts[10],
            pos_cell_idx=1,
            neg_cell_idx=0,
            curve_type="circular_arc_deg",
            theta0=300,
        )
    )
    edges.append(Edge(Verts[10], Verts[8], pos_cell_idx=1, neg_cell_idx=0))
    edges.append(
        Edge(
            Verts[11],
            Verts[11],
            pos_cell_idx=2,
            neg_cell_idx=1,
            curve_type="circle",
            radius=0.25 * PACMAN_SCALE,
        )
    )

    # dots
    edges.append(
        Edge(
            Verts[12],
            Verts[12],
            pos_cell_idx=4,
            neg_cell_idx=3,
            curve_type="circle",
            radius=0.1,
        )
    )
    edges.append(
        Edge(
            Verts[13],
            Verts[13],
            pos_cell_idx=5,
            neg_cell_idx=3,
            curve_type="circle",
            radius=0.1,
        )
    )
    edges.append(
        Edge(
            Verts[14],
            Verts[14],
            pos_cell_idx=6,
            neg_cell_idx=3,
            curve_type="circle",
            radius=0.1,
        )
    )

    # ghost
    edges.append(
        Edge(
            Verts[15],
            Verts[16],
            pos_cell_idx=8,
            neg_cell_idx=7,
            curve_type="sine_wave",
            amp=0.1,
            freq=6,
        )
    )
    edges.append(Edge(Verts[16], Verts[17], pos_cell_idx=8, neg_cell_idx=7))
    edges.append(
        Edge(
            Verts[17],
            Verts[18],
            pos_cell_idx=8,
            neg_cell_idx=7,
            curve_type="circular_arc_deg",
            theta0=180,
        )
    )
    edges.append(Edge(Verts[18], Verts[15], pos_cell_idx=8, neg_cell_idx=7))
    edges.append(
        Edge(
            Verts[19],
            Verts[19],
            pos_cell_idx=9,
            neg_cell_idx=8,
            curve_type="ellipse",
            a=0.15 * GHOST_SCALE,
            b=0.2 * GHOST_SCALE,
        )
    )
    edges.append(
        Edge(
            Verts[20],
            Verts[20],
            pos_cell_idx=10,
            neg_cell_idx=8,
            curve_type="ellipse",
            a=0.15 * GHOST_SCALE,
            b=0.2 * GHOST_SCALE,
        )
    )

    return edges
