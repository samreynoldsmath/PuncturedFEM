"""
PuncturedFEM
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
    "plot_trace",
    "plot_trace_log",
    "global_function_space",
    "bilinear_form",
    "solver",
]
