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
solver
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

from .locfun.locfun import locfun
from .locfun.locfunsp import locfunspace
from .locfun.nystrom import nystrom_solver
from .locfun.poly.poly import polynomial
from .mesh import meshlib
from .mesh.cell import cell
from .mesh.edge import edge
from .mesh.meshlib.__builder__ import mesh_builder
from .mesh.planar_mesh import planar_mesh
from .mesh.quad import quad
from .mesh.vert import vert
from .plot.edges import plot_edges
from .plot.plot_global_solution import plot_linear_combo
from .plot.traceplot import plot_trace, plot_trace_log
from .solver.bilinear_form import bilinear_form
from .solver.globfunsp import global_function_space
from .solver.solver import solver

__all__ = [
    "locfun",
    "locfunspace",
    "polynomial",
    "nystrom_solver",
    "cell",
    "edge",
    "meshlib",
    "planar_mesh",
    "quad",
    "vert",
    "mesh_builder",
    "plot_edges",
    "plot_linear_combo",
    "plot_trace",
    "plot_trace_log",
    "global_function_space",
    "bilinear_form",
    "solver",
]
