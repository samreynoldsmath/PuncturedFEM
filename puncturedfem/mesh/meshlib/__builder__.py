"""
__builder__.py
==============

Module containing the mesh builder function, which serves as template for the
mesh builder functions in the meshlib submodules.
"""

from typing import Any, Callable

from ..edge import Edge
from ..planar_mesh import PlanarMesh
from ..vert import Vert


def mesh_builder(
    get_verts: Callable[..., list[Vert]],
    get_edges: Callable[..., list[Edge]],
    verbose: bool = True,
    **kwargs: Any
) -> PlanarMesh:
    """
    Returns a planar mesh object given functions that define the Vertices and
    edges of the mesh.

    Parameters
    ----------
    get_verts : Callable
        Function that returns a list of Vertices.
    get_edges : Callable
        Function that returns a list of edges.
    verbose : bool, optional
        Whether to print information about the mesh, by default True.
    **kwargs : Any
        Additional keyword arguments to pass to get_verts and get_edges.

    Returns
    -------
    PlanarMesh
        Planar mesh object with specified Vertices and edges.
    """

    # define Vertices
    verts = get_verts(**kwargs)

    # TODO: set vertex ids here or in PlanarMesh constructor?
    for k, v in enumerate(verts):
        v.set_idx(k)

    # define edges
    edges = get_edges(verts, **kwargs)

    # return planar mesh
    return PlanarMesh(edges, verbose=verbose)
