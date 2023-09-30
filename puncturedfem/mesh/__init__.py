"""
mesh
====

Package for creating and manipulating meshes.

Modules
-------
bounding_box
    Contains the bounding_box function for computing bounding boxes.
cell
    Contains the cell class for creating mesh cells.
closed_contour
    Contains the closed_contour class for creating closed contours.
edge
    Contains the edge class for creating mesh edges.
planar_mesh
    Contains the planar_mesh class for creating planar meshes.
quad
    Contains the quad class for creating quadrature rules.
vert
    Contains the vert class for creating mesh vertices.

Libraries
---------
edgelib
    A library of modules for creating mesh edges.
meshlib
    A library of modules for creating meshes.
"""

from .cell import cell
from .edge import edge
from .planar_mesh import planar_mesh
from .quad import quad
from .vert import vert

__all__ = [
    "cell",
    "edge",
    "planar_mesh",
    "quad",
    "vert",
]
