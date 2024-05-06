"""
Template for the mesh builder functions in the meshlib submodules.

Routines in this module
-----------------------
mesh_builder

Notes
-----
- This module is likely to be replaced by a file i/o system in future versions.
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
    Build a planar mesh from a list of vertices and edges.

    This function is a template for the mesh builder functions in the meshlib
    submodules. It is used to build a planar mesh from a list of vertices and
    edges. The vertices and edges are obtained from the get_verts and get_edges
    functions, respectively.

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

    Notes
    -----
    - This function is likely to be replaced by a file i/o system in future
      versions.
    """
    # define vertices
    verts = get_verts(**kwargs)

    # set vertex indices
    for k, v in enumerate(verts):
        v.set_idx(k)

    # define edges
    edges = get_edges(verts, **kwargs)

    # return planar mesh
    return PlanarMesh(edges, verbose=verbose)
