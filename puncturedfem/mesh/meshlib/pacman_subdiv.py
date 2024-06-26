"""
Pac-Man and ghost mesh with internal vertical edges subdivided.

Routines in this module
-----------------------
pacman_subdiv
_get_verts
_get_edges
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
    Pac-Man chasing a ghost, with internal vertical edges subdivided.

    Parameters
    ----------
    verbose : bool, optional
        If True, print mesh information. Default is True.

    Returns
    -------
    PlanarMesh
        A planar mesh with the internal vertical edges subdivided.

    Notes
    -----
    - The mesh consists of 11 cells.
    """
    return mesh_builder(_get_verts, _get_edges, verbose=verbose)


def _get_verts() -> list[Vert]:
    # define Vertices
    verts: list[Vert] = []

    # rectangle corners
    verts.append(Vert(x=0.0, y=0.0))  # 0
    verts.append(Vert(x=1.0, y=0.0))  # 1
    verts.append(Vert(x=3.0, y=0.0))  # 2
    verts.append(Vert(x=4.0, y=0.0))  # 3
    verts.append(Vert(x=4.0, y=1.0))  # 4
    verts.append(Vert(x=3.0, y=1.0))  # 5
    verts.append(Vert(x=1.0, y=1.0))  # 6
    verts.append(Vert(x=0.0, y=1.0))  # 7

    # "Pac-Man"
    verts.append(Vert(x=0.5, y=0.5))  # 8
    verts.append(
        Vert(
            x=PACMAN_XSHIFT + PACMAN_SCALE * ROOT3OVER2,
            y=PACMAN_YSHIFT + PACMAN_SCALE * 0.5,
        )
    )  # 9
    verts.append(
        Vert(
            x=PACMAN_XSHIFT + PACMAN_SCALE * ROOT3OVER2,
            y=PACMAN_YSHIFT - PACMAN_SCALE * 0.5,
        )
    )  # 10
    verts.append(
        Vert(
            x=PACMAN_XSHIFT + PACMAN_SCALE * -0.1,
            y=PACMAN_YSHIFT + PACMAN_SCALE * 0.5,
        )
    )  # 11

    # central "dots"
    verts.append(Vert(x=1.5, y=0.5))  # 12
    verts.append(Vert(x=2.0, y=0.5))  # 13
    verts.append(Vert(x=2.5, y=0.5))  # 14

    # "ghost"
    verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (-0.5),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (-0.6),
        )
    )  # 15
    verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (0.5),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (-0.6),
        )
    )  # 16
    verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (0.5),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (0.2),
        )
    )  # 17
    verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (-0.5),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (0.2),
        )
    )  # 18
    verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (-0.25),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (0.1),
        )
    )  # 19
    verts.append(
        Vert(
            x=GHOST_X_SHIFT + GHOST_SCALE * (0.25),
            y=GHOST_Y_SHIFT + GHOST_SCALE * (0.1),
        )
    )  # 20

    # split vertical edges
    verts.append(Vert(x=1.0, y=0.5))  # 21
    verts.append(Vert(x=3.0, y=0.5))  # 22

    return verts


# EDGES ######################################################################


def _get_edges(verts: list[Vert]) -> list[Edge]:
    # define edges
    edges = []

    # rectangles
    edges.append(Edge(verts[0], verts[1], pos_cell_idx=0))
    edges.append(Edge(verts[1], verts[2], pos_cell_idx=3))
    edges.append(Edge(verts[2], verts[3], pos_cell_idx=7))
    edges.append(Edge(verts[3], verts[4], pos_cell_idx=7))
    edges.append(Edge(verts[4], verts[5], pos_cell_idx=7))
    edges.append(Edge(verts[5], verts[6], pos_cell_idx=3))
    edges.append(Edge(verts[6], verts[7], pos_cell_idx=0))
    edges.append(Edge(verts[7], verts[0], pos_cell_idx=0))
    edges.append(Edge(verts[1], verts[21], pos_cell_idx=0, neg_cell_idx=3))
    edges.append(Edge(verts[21], verts[6], pos_cell_idx=0, neg_cell_idx=3))
    edges.append(Edge(verts[2], verts[22], pos_cell_idx=3, neg_cell_idx=7))
    edges.append(Edge(verts[22], verts[5], pos_cell_idx=3, neg_cell_idx=7))

    # pacman
    edges.append(Edge(verts[8], verts[9], pos_cell_idx=1, neg_cell_idx=0))
    edges.append(
        Edge(
            verts[9],
            verts[10],
            pos_cell_idx=1,
            neg_cell_idx=0,
            curve_type="circular_arc_deg",
            theta0=300,
        )
    )
    edges.append(Edge(verts[10], verts[8], pos_cell_idx=1, neg_cell_idx=0))
    edges.append(
        Edge(
            verts[11],
            verts[11],
            pos_cell_idx=2,
            neg_cell_idx=1,
            quad_type="trap",
            curve_type="circle",
            radius=0.25 * PACMAN_SCALE,
        )
    )

    # dots
    edges.append(
        Edge(
            verts[12],
            verts[12],
            pos_cell_idx=4,
            neg_cell_idx=3,
            quad_type="trap",
            curve_type="circle",
            radius=0.1,
        )
    )
    edges.append(
        Edge(
            verts[13],
            verts[13],
            pos_cell_idx=5,
            neg_cell_idx=3,
            quad_type="trap",
            curve_type="circle",
            radius=0.1,
        )
    )
    edges.append(
        Edge(
            verts[14],
            verts[14],
            pos_cell_idx=6,
            neg_cell_idx=3,
            quad_type="trap",
            curve_type="circle",
            radius=0.1,
        )
    )

    # ghost
    edges.append(
        Edge(
            verts[15],
            verts[16],
            pos_cell_idx=8,
            neg_cell_idx=7,
            curve_type="sine_wave",
            amp=0.1,
            freq=6,
        )
    )
    edges.append(Edge(verts[16], verts[17], pos_cell_idx=8, neg_cell_idx=7))
    edges.append(
        Edge(
            verts[17],
            verts[18],
            pos_cell_idx=8,
            neg_cell_idx=7,
            curve_type="circular_arc_deg",
            theta0=180,
        )
    )
    edges.append(Edge(verts[18], verts[15], pos_cell_idx=8, neg_cell_idx=7))
    edges.append(
        Edge(
            verts[19],
            verts[19],
            pos_cell_idx=9,
            neg_cell_idx=8,
            quad_type="trap",
            curve_type="ellipse",
            a=0.15 * GHOST_SCALE,
            b=0.2 * GHOST_SCALE,
        )
    )
    edges.append(
        Edge(
            verts[20],
            verts[20],
            pos_cell_idx=10,
            neg_cell_idx=8,
            quad_type="trap",
            curve_type="ellipse",
            a=0.15 * GHOST_SCALE,
            b=0.2 * GHOST_SCALE,
        )
    )

    return edges
