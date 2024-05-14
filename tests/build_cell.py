"""
cell_builder.py
===============

Build mesh cells for testing purposes.
"""

from typing import Any

import numpy as np

import puncturedfem as pf


def build_circle(
    radius: float = 0.25, center_x: float = 0.5, center_y: float = 0.5
) -> tuple[pf.MeshCell, dict[str, Any]]:
    """
    Construct a mesh cell that is a disk with radius 0.25 centered at (0.5, 0.5)
    """

    # define circle parameters
    cell_data = {
        "center": np.array([center_x, center_y]),
        "radius": radius,
    }
    cell_data["boundary_length"] = 2 * np.pi * radius

    # define central "vertex"
    verts: list[pf.Vert] = []
    verts.append(pf.Vert(x=center_x, y=center_y))

    # define edge
    edges: list[pf.Edge] = []
    edges.append(
        pf.Edge(
            verts[0],
            verts[0],
            pos_cell_idx=0,
            curve_type="circle",
            quad_type="trap",
            radius=cell_data["radius"],
        )
    )

    # define mesh cell
    K = pf.MeshCell(idx=0, edges=edges)

    return K, cell_data


def build_square(
    side_length: float = 1.0,
) -> tuple[pf.MeshCell, dict[str, float]]:
    """
    Construct a mesh cell that is a square with side length 1
    """

    cell_data = {"side_length": side_length}
    cell_data["boundary_length"] = 4 * side_length

    # define vertices
    verts: list[pf.Vert] = []
    verts.append(pf.Vert(x=0.0, y=0.0))
    verts.append(pf.Vert(x=cell_data["side_length"], y=0.0))
    verts.append(
        pf.Vert(x=cell_data["side_length"], y=cell_data["side_length"])
    )
    verts.append(pf.Vert(x=0.0, y=cell_data["side_length"]))

    # define edges
    edges: list[pf.Edge] = []
    edges.append(pf.Edge(verts[0], verts[1], pos_cell_idx=0))
    edges.append(pf.Edge(verts[1], verts[2], pos_cell_idx=0))
    edges.append(pf.Edge(verts[2], verts[3], pos_cell_idx=0))
    edges.append(pf.Edge(verts[3], verts[0], pos_cell_idx=0))

    # define mesh cell
    K = pf.MeshCell(idx=0, edges=edges)

    return K, cell_data


def build_punctured_square(
    side_length: float = 1.0,
    radius: float = 0.25,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> tuple[pf.MeshCell, dict[str, Any]]:
    """
    Construct a mesh cell that is a square with side length 1 with a disk of
    radius 0.25 centered at (0.5, 0.5) removed.
    """

    cell_data = {
        "side_length": side_length,
        "radius": radius,
        "center": np.array([center_x, center_y]),
    }
    cell_data["boundary_length"] = 4 * side_length + 2 * np.pi * radius

    # define vertices
    verts: list[pf.Vert] = []
    verts.append(pf.Vert(x=0.0, y=0.0))
    verts.append(pf.Vert(x=side_length, y=0.0))
    verts.append(pf.Vert(x=side_length, y=side_length))
    verts.append(pf.Vert(x=0.0, y=side_length))
    verts.append(pf.Vert(x=center_x, y=center_y))

    # define edges
    edges: list[pf.Edge] = []
    edges.append(pf.Edge(verts[0], verts[1], pos_cell_idx=0))
    edges.append(pf.Edge(verts[1], verts[2], pos_cell_idx=0))
    edges.append(pf.Edge(verts[2], verts[3], pos_cell_idx=0))
    edges.append(pf.Edge(verts[3], verts[0], pos_cell_idx=0))
    edges.append(
        pf.Edge(
            verts[4],
            verts[4],
            neg_cell_idx=0,
            curve_type="circle",
            quad_type="trap",
            radius=cell_data["radius"],
        )
    )

    # define mesh cell
    K = pf.MeshCell(idx=0, edges=edges)

    return K, cell_data


def build_ghost() -> pf.MeshCell:
    """Make the ghosty boi"""

    # define vertices
    verts = []
    verts.append(pf.Vert(x=0.0, y=0.0))  # 0
    verts.append(pf.Vert(x=1.0, y=0.0))  # 1
    verts.append(pf.Vert(x=1.0, y=0.8))  # 2
    verts.append(pf.Vert(x=0.0, y=0.8))  # 3
    verts.append(pf.Vert(x=0.25, y=0.7))  # 4
    verts.append(pf.Vert(x=0.75, y=0.7))  # 5

    # define edges
    edges = []
    edges.append(
        pf.Edge(
            verts[0],
            verts[1],
            pos_cell_idx=0,
            curve_type="sine_wave",
            amp=0.1,
            freq=6,
        )
    )
    edges.append(pf.Edge(verts[1], verts[2], pos_cell_idx=0))
    edges.append(
        pf.Edge(
            verts[2],
            verts[3],
            pos_cell_idx=0,
            curve_type="circular_arc_deg",
            theta0=180,
        )
    )
    edges.append(pf.Edge(verts[3], verts[0], pos_cell_idx=0))
    edges.append(
        pf.Edge(
            verts[4],
            verts[4],
            neg_cell_idx=0,
            curve_type="ellipse",
            a=0.15,
            b=0.2,
        )
    )
    edges.append(
        pf.Edge(
            verts[5],
            verts[5],
            neg_cell_idx=0,
            curve_type="ellipse",
            a=0.15,
            b=0.2,
        )
    )

    # define mesh cell
    return pf.MeshCell(idx=0, edges=edges)
