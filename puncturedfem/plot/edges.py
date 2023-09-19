import matplotlib.pyplot as plt
import numpy as np

from .. import mesh


def plot_edges(
    edge_list, orientation=False, axis_arg="equal", grid_arg="minor"
):
    plt.figure()
    plt.axis(axis_arg)
    plt.grid(grid_arg)

    for e in edge_list:
        if orientation:
            _plot_oriented_edge(e)
        else:
            _plot_edge(e)

    plt.show()

    return None


def plot_boundary(
    K: mesh.cell,
    orientation=False,
    hole_int_pts=False,
    axis_arg="equal",
    grid_arg="minor",
):
    plt.figure()
    plt.axis(axis_arg)
    plt.grid(grid_arg)

    for e in K.edge_list:
        if orientation:
            _plot_oriented_edge(e)
        else:
            _plot_edge(e)

    if hole_int_pts:
        _plot_hole_interior_points(K)

    plt.show()

    return None


def _plot_edge(e):
    plt.plot(e.x[0, :], e.x[1, :], "k-")


def _plot_oriented_edge(e):
    X = e.x[0, :]
    Y = e.x[1, :]
    U = np.roll(X, -1) - X
    V = np.roll(Y, -1) - Y
    X = X[:-1]
    Y = Y[:-1]
    U = U[:-1]
    V = V[:-1]
    plt.quiver(X, Y, U, V, scale=1, angles="xy", scale_units="xy")
    return None


def _plot_hole_interior_points(K):
    plt.scatter(K.hole_int_pts[0, :], K.hole_int_pts[1, :])
