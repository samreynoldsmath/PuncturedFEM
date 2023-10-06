"""
traceplot.py
============

Module containing functions for plotting traces of functions on the boundary of
a MeshCell.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..mesh.cell import MeshCell
from ..mesh.quad import Quad

PI_CHAR = r"$\pi$"


def plot_trace(
    f_trace_list: list[np.ndarray],
    fmt: str,
    legend: str,
    title: str,
    K: MeshCell,
    quad_list: list[Quad],
) -> None:
    """
    Plots the trace of a function on the boundary of a MeshCell.
    """

    t = _get_trace_param_mesh_boundary(K, quad_list)

    plt.figure()
    for k, f_trace in enumerate(f_trace_list):
        plt.plot(t, f_trace, fmt[k])
    plt.legend(legend)
    plt.grid(True)
    plt.title(title)

    x_ticks, x_labels = _get_ticks(K)
    plt.xticks(ticks=x_ticks, labels=x_labels)

    plt.show()


def plot_trace_log(
    f_trace_list: list[np.ndarray],
    fmt: str,
    legend: str,
    title: str,
    K: MeshCell,
    quad_list: list[Quad],
) -> None:
    """
    Plots the trace of a function on the boundary of a MeshCell with a log scale
    on the vertical axis.
    """

    t = _get_trace_param_mesh_boundary(K, quad_list)

    plt.figure()
    for k, f_trace in enumerate(f_trace_list):
        plt.semilogy(t, f_trace, fmt[k])
    plt.legend(legend)
    plt.grid(True)
    plt.title(title)

    x_ticks, x_labels = _get_ticks(K)
    plt.xticks(ticks=x_ticks, labels=x_labels)

    plt.show()


def _make_quad_dict(quad_list: list[Quad]) -> dict[str, Quad]:
    """
    Organize a list of distinct Quad objects into a convenient dictionary
    """
    quad_dict = {}
    for q in quad_list:
        quad_dict[q.type] = q
    return quad_dict


def _get_trace_param_mesh_boundary(
    K: MeshCell, quad_list: list[Quad]
) -> np.ndarray:
    """
    Returns the trace parameter values for the boundary of a MeshCell.
    """

    quad_dict = _make_quad_dict(quad_list)

    t = np.zeros((K.num_pts,))
    t0 = 0.0
    idx_start = 0
    for c in K.components:
        for e in c.edges:
            t[idx_start : (idx_start + e.num_pts - 1)] = (
                t0 + quad_dict[e.quad_type].t[:-1]
            )
            idx_start += e.num_pts - 1
            t0 += 2 * np.pi

    return t


def _get_ticks(K: MeshCell) -> tuple[np.ndarray, list[str]]:
    """
    Returns the x-ticks and x-labels for a plot of a function on the boundary of
    a MeshCell.
    """
    x_ticks = np.linspace(0, 2 * np.pi * K.num_edges, K.num_edges + 1)
    x_labels = [
        "0",
    ]
    for k in range(1, K.num_edges + 1):
        x_labels.append(f"{2 * k}{PI_CHAR}")
    return x_ticks, x_labels
