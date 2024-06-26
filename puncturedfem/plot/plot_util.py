"""
Utility functions for plotting.

Functions
---------
save_figure(filename, dpi=300, bbox_inches="tight")
    Save a figure to a file.
get_axis_limits(edges, pad=0.1)
    Get the axis limits for a list of edges.
get_figure_size(min_x, max_x, min_y, max_y, h=4.0)
    Get the figure size, returning the width and height.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ..mesh.edge import Edge


def save_figure(
    filename: str, dpi: int = 300, bbox_inches: str = "tight"
) -> None:
    """
    Save a figure to a file.

    Parameters
    ----------
    filename : str
        The name of the file to save the figure to.
    dpi : int, optional
        The resolution of the figure. The default is 300.
    bbox_inches : str, optional
        The bounding box in inches. The default is "tight".
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)


def get_axis_limits(
    edges: list[Edge], pad: float = 0.1
) -> tuple[float, float, float, float]:
    """
    Get the axis limits for a list of edges.

    Parameters
    ----------
    edges : list of Edge
        The edges to determine the axis limits from.
    pad : float, optional
        The padding to add to the axis limits. The default is 0.1.

    Returns
    -------
    min_x : float
        The minimum x-value.
    max_x : float
        The maximum x-value.
    min_y : float
        The minimum y-value.
    max_y : float
        The maximum y-value.
    """
    # initial values
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf

    # update values
    for e in edges:
        min_x = _update_min(min_x, e.x[0, :])
        max_x = _update_max(max_x, e.x[0, :])
        min_y = _update_min(min_y, e.x[1, :])
        max_y = _update_max(max_y, e.x[1, :])

    # add padding
    if pad > 0.0:
        dx = max_x - min_x
        dy = max_y - min_y
        min_x -= pad * dx
        max_x += pad * dx
        min_y -= pad * dy
        max_y += pad * dy

    # return window
    return min_x, max_x, min_y, max_y


def get_figure_size(
    min_x: float, max_x: float, min_y: float, max_y: float, h: float = 4.0
) -> tuple[float, float]:
    """
    Get the height and width of the figure.

    Parameters
    ----------
    min_x : float
        The minimum x-value.
    max_x : float
        The maximum x-value.
    min_y : float
        The minimum y-value.
    max_y : float
        The maximum y-value.
    h : float, optional
        The height of the figure. The default is 4.0.

    Returns
    -------
    w : float
        The width of the figure.
    h : float
        The height of the figure.
    """
    dx = max_x - min_x
    dy = max_y - min_y
    w = h * dx / dy
    return w, h


def _update_min(current_min: float, candidates: np.ndarray) -> float:
    min_candidate = min(candidates)
    return min(current_min, min_candidate)


def _update_max(current_max: float, candidates: np.ndarray) -> float:
    max_candidate = max(candidates)
    return max(current_max, max_candidate)
