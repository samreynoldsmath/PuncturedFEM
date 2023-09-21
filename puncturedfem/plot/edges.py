import matplotlib.pyplot as plt
import numpy as np

from ..mesh.cell import cell
from ..mesh.edge import edge


def plot_edges(
    edge_list: list[edge],
    orientation: bool = False,
    axis_arg: str = "equal",
    grid_arg: bool = True,
) -> None:
    plt.figure()
    plt.axis(axis_arg)
    plt.grid(grid_arg)

    for e in edge_list:
        if orientation:
            _plot_oriented_edge(e)
        else:
            _plot_edge(e)

    plt.show()


def plot_boundary(
    K: cell,
    orientation: bool = False,
    hole_int_pts: bool = False,
    axis_arg: str = "equal",
    grid_arg: bool = True,
) -> None:
    plt.figure()
    plt.axis(axis_arg)
    plt.grid(grid_arg)

    for e in K.get_edges():
        if orientation:
            _plot_oriented_edge(e)
        else:
            _plot_edge(e)

    if hole_int_pts:
        _plot_hole_interior_points(K)

    plt.show()


def _plot_edge(e: edge) -> None:
    plt.plot(e.x[0, :], e.x[1, :], "k-")


def _plot_oriented_edge(e: edge) -> None:
    X = e.x[0, :]
    Y = e.x[1, :]
    U = np.roll(X, -1) - X
    V = np.roll(Y, -1) - Y
    X = X[:-1]
    Y = Y[:-1]
    U = U[:-1]
    V = V[:-1]
    plt.quiver(X, Y, U, V, scale=1, angles="xy", scale_units="xy")


def _plot_hole_interior_points(K: cell) -> None:
    plt.scatter(K.int_x1, K.int_x2)
