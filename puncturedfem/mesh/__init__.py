"""
Create and manipulate meshes.

Modules
-------
bounding_box
    Contains the bounding_box function for computing bounding boxes.
cell
    Contains the MeshCell class for creating mesh cells.
closed_contour
    Contains the ClosedContour class for creating closed contours.
edge
    Contains the Edge class for creating mesh edges.
mesh_exceptions
    Contains exceptions for the mesh module.
planar_mesh
    Contains the PlanarMesh class for creating planar meshes.
quad
    Contains the Quad class for creating quadrature rules.
split_edge
    Contains the split_edge function for splitting edges.
transform
    Contains tools for transforming edges.
vert
    Contains the Vert class for creating mesh vertices.

Libraries
---------
edgelib
    A library of modules for creating mesh edges.
meshlib
    A library of modules for creating meshes.
"""
from .cell import MeshCell
from .edge import Edge
from .planar_mesh import PlanarMesh
from .quad import Quad
from .vert import Vert

__all__ = [
    "MeshCell",
    "Edge",
    "PlanarMesh",
    "Quad",
    "Vert",
]
