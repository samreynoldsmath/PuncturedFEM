"""
PuncturedFEM
============

A Python package for the numerical solution of PDEs on meshes with punctured
and curvilinear elements.

The method can be roughly described as a "non-virtual" virtual element method,
as the same function space is used, but the basis functions are used directly
as degrees of freedom. The method also borrows ideas from boundary element
methods, specifically when it comes to performing computations (e.g. H^1
semi-inner products) with the implicitly-defined basis functions.

See README.md for the latest references and more information.


Usage
-----
See examples/ for usage examples.


Subpackages
-----------
locfun
    Local function space and basis functions.
mesh
    Mesh data structures and mesh generation.
plot
    Plotting functions.
Solver
    Solver for PDEs on punctured meshes.


License
-------
Copyright (C) 2022 - 2023 Jeffrey S. Ovall and Samuel E. Reynolds.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see https://www.gnu.org/licenses/.
"""

from . import plot
from .locfun.edge_space import EdgeSpace
from .locfun.locfun import LocalFunction
from .locfun.locfunsp import LocalFunctionSpace
from .locfun.nystrom import NystromSolver
from .locfun.poly.barycentric import (
    barycentric_coordinates,
    barycentric_coordinates_edge,
)
from .locfun.poly.piecewise_poly import PiecewisePolynomial
from .locfun.poly.poly import Polynomial
from .locfun.trace import DirichletTrace
from .mesh import meshlib
from .mesh.cell import MeshCell
from .mesh.edge import Edge
from .mesh.meshlib.__builder__ import mesh_builder
from .mesh.planar_mesh import PlanarMesh
from .mesh.quad import Quad, QuadDict, get_quad_dict
from .mesh.split_edge import split_edge
from .mesh.vert import Vert
from .solver.bilinear_form import BilinearForm
from .solver.globfunsp import GlobalFunctionSpace
from .solver.solver import Solver

__all__ = [
    # mesh
    "Edge",
    "MeshCell",
    "PlanarMesh",
    "Quad",
    "QuadDict",
    "Vert",
    "get_quad_dict",
    "meshlib",
    "mesh_builder",
    "split_edge",

    # locfun
    "DirichletTrace",
    "EdgeSpace",
    "LocalFunction",
    "LocalFunctionSpace",
    "NystromSolver",
    "Polynomial",
    "PiecewisePolynomial",
    "barycentric_coordinates",
    "barycentric_coordinates_edge",

    # solver
    "GlobalFunctionSpace",
    "BilinearForm",
    "Solver",

    # plot
    "plot",
]
