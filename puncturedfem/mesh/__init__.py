"""
mesh
====

Package for creating and manipulating meshes.

Modules
-------
bounding_box
    Contains the bounding_box function for computing bounding boxes.
MeshCell
    Contains the MeshCell class for creating mesh cells.
ClosedContour
    Contains the ClosedContour class for creating closed contours.
Edge
    Contains the Edge class for creating mesh edges.
PlanarMesh
    Contains the PlanarMesh class for creating planar meshes.
Quad
    Contains the Quad class for creating Quadrature rules.
Vert
    Contains the Vert class for creating mesh Vertices.

Libraries
---------
Edgelib
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
