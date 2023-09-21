from ..edge import edge
from ..planar_mesh import planar_mesh
from ..vert import vert
from .__builder__ import mesh_builder


def square_circular_hole(verbose: bool = True) -> planar_mesh:
    return mesh_builder(get_verts, get_edges, verbose=verbose)


# VERTICES ###################################################################


def get_verts() -> list[vert]:
    verts: list[vert] = []
    verts.append(vert(x=0.0, y=0.0))
    verts.append(vert(x=2.0, y=0.0))
    verts.append(vert(x=2.0, y=1.0))
    verts.append(vert(x=0.0, y=1.0))
    verts.append(vert(x=0.5, y=0.5))  # center of circle
    return verts


def get_edges(verts: list[vert]) -> list[edge]:
    edges: list[edge] = []
    edges.append(edge(verts[0], verts[1], pos_cell_idx=0))
    edges.append(edge(verts[1], verts[2], pos_cell_idx=0))
    edges.append(edge(verts[2], verts[3], pos_cell_idx=0))
    edges.append(edge(verts[3], verts[0], pos_cell_idx=0))
    edges.append(
        edge(
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
