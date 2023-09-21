from typing import Any, Callable

from ..planar_mesh import planar_mesh


def mesh_builder(
    get_verts: Callable,
    get_edges: Callable,
    verbose: bool = True,
    **kwargs: Any
) -> planar_mesh:
    # define vertices
    verts = get_verts(**kwargs)

    # TODO: set vertex ids here?
    for k, v in enumerate(verts):
        v.set_id(k)

    # define edges
    edges = get_edges(verts, **kwargs)

    # return planar mesh
    return planar_mesh(edges, verbose=verbose)
