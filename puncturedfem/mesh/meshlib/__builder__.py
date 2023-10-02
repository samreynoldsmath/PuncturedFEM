"""
__builder__.py
==============

Module containing the mesh builder function, which serves as template for the
mesh builder functions in the meshlib submodules.
"""

from typing import Any, Callable

from ..edge import edge
from ..planar_mesh import planar_mesh
from ..vert import vert


def mesh_builder(
    get_verts: Callable[..., list[vert]],
    get_edges: Callable[..., list[edge]],
    verbose: bool = True,
    **kwargs: Any
) -> planar_mesh:
    """
    Returns a planar mesh object given functions that define the vertices and
    edges of the mesh.

    Parameters
    ----------
    get_verts : Callable
        Function that returns a list of vertices.
    get_edges : Callable
        Function that returns a list of edges.
    verbose : bool, optional
        Whether to print information about the mesh, by default True.
    **kwargs : Any
        Additional keyword arguments to pass to get_verts and get_edges.

    Returns
    -------
    planar_mesh
        Planar mesh object with specified vertices and edges.
    """

    # define vertices
    verts = get_verts(**kwargs)

    # TODO: set vertex ids here or in planar_mesh constructor?
    for k, v in enumerate(verts):
        v.set_idx(k)

    # define edges
    edges = get_edges(verts, **kwargs)

    # return planar mesh
    return planar_mesh(edges, verbose=verbose)
