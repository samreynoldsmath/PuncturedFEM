"""
Rigid motions and dilations in the plane.

Functions
---------
join_points(x, a, b)
    Join the points a to b.
rotate(x, theta)
    Rotate counterclockwise by theta (degrees).
reflect_across_x_axis(x)
    Reflect across the horizontal axis.
reflect_across_y_axis(x)
    Reflect across the vertical axis.
translate(x, a)
    Translate by a vector a.
dilate(x, alpha)
    Dilate by a scalar alpha.
apply_orthogonal_transformation(x, A)
    Apply an orthogonal transformation to x.
"""

import numpy as np

from .mesh_exceptions import EdgeTransformationError
from .vert import Vert

TOL = 1e-12


def join_points(x: np.ndarray, a: Vert, b: Vert) -> np.ndarray:
    """
    Join the points a to b.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (2, n) representing the points on the curve to be transformed.
    a : Vert
        Starting point.
    b : Vert
        Ending point.

    Returns
    -------
    np.ndarray
        Array of shape (2, n) representing the transformed points.
    """
    # check that specified endpoints are distinct
    ab_norm = np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    if ab_norm < TOL:
        raise EdgeTransformationError("a and b must be distinct points")

    # check that endpoints of sampled points are distinct
    xi = x[:, 0]
    xf = x[:, -1]
    xi_xf_norm = np.sqrt((xi[0] - xf[0]) ** 2 + (xi[1] - xf[1]) ** 2)
    if xi_xf_norm < TOL:
        raise EdgeTransformationError("Edge must have distinct endpoints")

    # anchor starting point to origin
    x = translate(x, Vert(x=-xi[0], y=-xi[1]))

    # rotate
    theta = -np.arctan2(xf[1] - xi[1], xf[0] - xi[0])
    theta += np.arctan2(b.y - a.y, b.x - a.x)
    theta *= 180 / np.pi
    x = rotate(x, theta)

    # rescale
    alpha = ab_norm / xi_xf_norm
    x = dilate(x, alpha)

    # anchor at point a
    return translate(x, a)


def rotate(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate counterclockwise by theta (degrees).

    Parameters
    ----------
    x : np.ndarray
        Array of shape (2, n) representing the points on the curve to be transformed.
    theta : float
        Angle of rotation in degrees.

    Returns
    -------
    np.ndarray
        Array of shape (2, n) representing the transformed points.
    """
    c = np.cos(theta * np.pi / 180)
    s = np.sin(theta * np.pi / 180)
    R = np.array([[c, -s], [s, c]])
    return apply_orthogonal_transformation(x, R)


def reflect_across_x_axis(x: np.ndarray) -> np.ndarray:
    """
    Reflect across the horizontal axis.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (2, n) representing the points on the curve to be transformed.

    Returns
    -------
    np.ndarray
        Array of shape (2, n) representing the transformed points.
    """
    A = np.array([[1, 0], [0, -1]])
    return apply_orthogonal_transformation(x, A)


def reflect_across_y_axis(x: np.ndarray) -> np.ndarray:
    """
    Reflect across the vertical axis.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (2, n) representing the points on the curve to be transformed.

    Returns
    -------
    np.ndarray
        Array of shape (2, n) representing the transformed points.
    """
    A = np.array([[-1, 0], [0, 1]])
    return apply_orthogonal_transformation(x, A)


def translate(x: np.ndarray, a: Vert) -> np.ndarray:
    """
    Translate by a vector a.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (2, n) representing the points on the curve to be transformed.
    a : Vert
        Translation vector.

    Returns
    -------
    np.ndarray
        Array of shape (2, n) representing the transformed points.
    """
    x[0, :] += a.x
    x[1, :] += a.y
    return x


def dilate(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Dilate by a scalar alpha.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (2, n) representing the points on the curve to be transformed.
    alpha : float
        Dilation factor.

    Returns
    -------
    np.ndarray
        Array of shape (2, n) representing the transformed points.
    """
    if np.abs(alpha) < TOL:
        raise EdgeTransformationError("Dilation factor alpha must be nonzero")
    return alpha * x


def apply_orthogonal_transformation(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Apply an orthogonal transformation to x.

    Transforms 2-dimensional space with the linear map
            x mapsto A * x
    where A is a 2 by 2 orthogonal matrix, i.e. A^T * A = I It is important that
    A is orthogonal, since the first derivative norm as well as the curvature
    are invariant under such a transformation.

    Parameters
    ----------
    A : np.ndarray
        2 by 2 orthogonal matrix.
    x : np.ndarray
        Array of shape (2, n) representing the points on the curve to be transformed.

    Returns
    -------
    np.ndarray
        Array of shape (2, n) representing the transformed points.
    """
    # safety checks
    msg = "A must be a 2 by 2 orthogonal matrix"
    if np.shape(A) != (2, 2):
        raise EdgeTransformationError(msg)
    if np.linalg.norm(np.transpose(A) @ A - np.eye(2)) > TOL:
        raise EdgeTransformationError(msg)

    # apply transformation to vector quantities
    return A @ x
