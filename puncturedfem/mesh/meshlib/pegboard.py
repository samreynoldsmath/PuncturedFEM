from ..edge import Edge
from ..planar_mesh import PlanarMesh
from ..vert import Vert
from .__builder__ import mesh_builder


def pegboard(
    size: tuple[int, int], radius: float = 0.25, verbose: bool = True
) -> PlanarMesh:
    return mesh_builder(
        _get_verts, _get_edges, verbose=verbose, size=size, radius=radius
    )


def _get_verts(size: tuple[int, int], radius: float) -> list[Vert]:
    verts: list[Vert] = []

    m = size[0] + 1
    n = size[1] + 1

    # rescale so that the longest edge is 1
    h = 1 / max(m - 1, n - 1)

    for i in range(m):
        for j in range(n):
            verts.append(Vert(x=h * j, y=h * i))  # index = i * n + j

    # circular holes
    if radius <= 0:
        return verts

    for i in range(m - 1):
        for j in range(n - 1):
            verts.append(Vert(x=h * (j + 0.5), y=h * (i + 0.5)))

    return verts


# EDGES ######################################################################


def _get_edges(
    verts: list[Vert], size: tuple[int, int], radius: float
) -> list[Edge]:
    edges: list[Edge] = []

    m = size[0] + 1
    n = size[1] + 1

    h = 1 / max(m - 1, n - 1)

    for i in range(m - 1):
        for j in range(n - 1):
            cell_idx = i * (n - 1) + j
            if j == 0:
                left_idx = -1
            else:
                left_idx = cell_idx - 1
            if i == 0:
                bottom_idx = -1
            else:
                bottom_idx = cell_idx - (n - 1)
            # horizontal edges
            edges.append(
                Edge(
                    verts[i * n + j],
                    verts[i * n + j + 1],
                    pos_cell_idx=cell_idx,
                    neg_cell_idx=bottom_idx,
                )
            )
            # vertical edges
            edges.append(
                Edge(
                    verts[i * n + j],
                    verts[(i + 1) * n + j],
                    pos_cell_idx=left_idx,
                    neg_cell_idx=cell_idx,
                )
            )
        # final vertical edge
        edges.append(
            Edge(
                verts[i * n + n - 1],
                verts[(i + 1) * n + n - 1],
                pos_cell_idx=cell_idx,
            )
        )
    # final horizontal edges
    for j in range(n - 1):
        bottom_idx = (m - 2) * (n - 1) + j
        edges.append(
            Edge(
                verts[(m - 1) * n + j],
                verts[(m - 1) * n + j + 1],
                neg_cell_idx=bottom_idx,
            )
        )

    # circular holes
    if radius <= 0:
        return edges

    for i in range(m - 1):
        for j in range(n - 1):
            neg_cell_idx = i * (n - 1) + j
            pos_cell_idx = (m - 1) * (n - 1) + neg_cell_idx
            vert_idx = m * n + i * (n - 1) + j
            edges.append(
                Edge(
                    verts[vert_idx],
                    verts[vert_idx],
                    pos_cell_idx=pos_cell_idx,
                    neg_cell_idx=neg_cell_idx,
                    quad_type="trap",
                    curve_type="circle",
                    radius=radius * h,
                )
            )

    return edges
