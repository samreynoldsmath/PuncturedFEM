"""
edge_param.py
=======

Module containing the ParameterizedEdge class.
"""

from typing import Any, Callable

import numpy as np

from .bounding_box import get_bounding_box
from .mesh_exceptions import EdgeTransformationError, SizeMismatchError
from .quad import Quad
from .vert import Vert


class ParameterizedEdge:
    anchor: Vert
    endpnt: Vert
    is_loop: bool
    num_pts: int
    interp: int
    x: np.ndarray
    unit_tangent: np.ndarray
    unit_normal: np.ndarray
    dx_norm: np.ndarray
    curvature: np.ndarray

    def __init__(
        self, anchor: Vert, endpnt: Vert, q: Quad, gamma: Any, **curve_opts: Any
    ) -> None:
        """
        Parameterize the Edge using the specified Quadrature rule. The
        parameterization is stored in the following attributes:
            x : np.ndarray
                The sampled points on the Edge.
            unit_tangent : np.ndarray
                The unit tangent vector at each sampled point on the Edge.
            unit_normal : np.ndarray
                The unit normal vector at each sampled point on the Edge.
            dx_norm : np.ndarray
                The norm of the derivative of the parameterization at each
                sampled point on the Edge.
            curvature : np.ndarray
                The signed curvature at each sampled point on the Edge.
        """

        # set vertices
        self.set_verts(anchor, endpnt)

        # store parameterization type
        self.quad_type = q.type

        # 2 * n + 1 points sampled per Edge
        self.num_pts = 2 * q.n + 1

        # points on the boundary
        self.x = gamma.X(q.t, **curve_opts)

        # unweighted square norm of derivative
        dx = gamma.DX(q.t, **curve_opts)
        dx2 = dx[0, :] ** 2 + dx[1, :] ** 2

        # norm of derivative (with chain rule)
        self.dx_norm = np.sqrt(dx2) * q.wgt

        # unit tangent vector
        self.unit_tangent = dx / np.sqrt(dx2)

        # outward unit normal vector
        self.unit_normal = np.zeros((2, self.num_pts))
        self.unit_normal[0, :] = self.unit_tangent[1, :]
        self.unit_normal[1, :] = -self.unit_tangent[0, :]

        # signed curvature
        ddx = gamma.DDX(q.t, **curve_opts)
        self.curvature = (
            ddx[0, :] * self.unit_normal[0, :]
            + ddx[1, :] * self.unit_normal[1, :]
        ) / dx2

        # set endpoints
        if self.is_loop:
            self.translate(z=self.anchor)
        else:
            self.join_points(a=self.anchor, b=self.endpnt)

    def set_verts(self, anchor: Vert, endpnt: Vert) -> None:
        """Set the anchor and endpnt Vertices of the Edge"""
        self.anchor = anchor
        self.endpnt = endpnt
        self.is_loop = self.anchor == self.endpnt

    def get_sampled_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the sampled points on the Edge"""
        return self.x[0, :], self.x[1, :]

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """Return the bounding box of the Edge"""
        return get_bounding_box(x=self.x[0, :], y=self.x[1, :])

    # TRANSFORMATIONS ########################################################

    def reverse_orientation(self) -> None:
        """
        Reverse the orientation of this Edge using the reparameterization
        x(2 pi - t). The chain rule flips the sign of some derivative-based
        quantities.
        """

        # vector quantities
        self.x = np.fliplr(self.x)
        self.unit_tangent = -np.fliplr(self.unit_tangent)
        self.unit_normal = -np.fliplr(self.unit_normal)

        # scalar quantities
        self.dx_norm = np.flip(self.dx_norm)
        self.curvature = -np.flip(self.curvature)

        # set vertices
        self.set_verts(self.endpnt, self.anchor)

    def join_points(self, a: Vert, b: Vert) -> None:
        """Join the points a to b with this Edge."""

        # tolerance for floating point comparisons
        TOL = 1e-12

        # check that specified endpoints are distinct
        ab_norm = np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
        if ab_norm < TOL:
            raise EdgeTransformationError("a and b must be distinct points")

        # check that endpoints of Edge are distinct
        x = self.x[:, 0]
        y = self.x[:, self.num_pts - 1]
        xy_norm = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
        if xy_norm < TOL:
            raise EdgeTransformationError("Edge must have distinct endpoints")

        # anchor starting point to origin
        self.translate(Vert(x=-x[0], y=-x[1]))

        # rotate
        theta = -np.arctan2(y[1] - x[1], y[0] - x[0])
        theta += np.arctan2(b.y - a.y, b.x - a.x)
        theta *= 180 / np.pi
        self.rotate(theta)

        # rescale
        alpha = ab_norm / xy_norm
        self.dilate(alpha)

        # anchor at point a
        self.translate(a)

    def translate(self, z: Vert) -> None:
        """Translate by a vector z"""

        self.x[0, :] += z.x
        self.x[1, :] += z.y
        self.set_verts(self.anchor + z, self.endpnt + z)

    def dilate(self, alpha: float) -> None:
        """Dilate by a scalar alpha"""

        # tolerance for floating point comparisons
        TOL = 1e-12

        if np.abs(alpha) < TOL:
            raise EdgeTransformationError(
                "Dilation factor alpha must be nonzero"
            )

        self.x *= alpha
        self.dx_norm *= np.abs(alpha)
        self.curvature *= 1 / alpha

        self.set_verts(alpha * self.anchor, alpha * self.endpnt)

    def rotate(self, theta: float) -> None:
        """Rotate counterclockwise by theta (degrees)"""

        if theta % 360 == 0:
            return

        c = np.cos(theta * np.pi / 180)
        s = np.sin(theta * np.pi / 180)
        R = np.array([[c, -s], [s, c]])

        self.apply_orthogonal_transformation(R)

    def reflect_across_x_axis(self) -> None:
        """Reflect across the horizontal axis"""

        A = np.array([[1, 0], [0, -1]])
        self.apply_orthogonal_transformation(A)

    def reflect_across_y_axis(self) -> None:
        """Reflect across the vertical axis"""
        A = np.array([[-1, 0], [0, 1]])
        self.apply_orthogonal_transformation(A)

    def apply_orthogonal_transformation(self, A: np.ndarray) -> None:
        """
        Transforms 2-dimensional space with the linear map
                x mapsto A * x
        where A is a 2 by 2 orthogonal matrix, i.e. A^T * A = I

        It is important that A is orthogonal, since the first derivative norm
        as well as the curvature are invariant under such a transformation.
        """

        # tolerance for floating point comparisons
        TOL = 1e-12

        # safety checks
        msg = "A must be a 2 by 2 orthogonal matrix"
        if np.shape(A) != (2, 2):
            raise EdgeTransformationError(msg)
        if np.linalg.norm(np.transpose(A) @ A - np.eye(2)) > TOL:
            raise EdgeTransformationError(msg)

        # apply transformation to vector quantities
        self.x = A @ self.x
        self.unit_tangent = A @ self.unit_tangent
        self.unit_normal[0, :] = self.unit_tangent[1, :]
        self.unit_normal[1, :] = -self.unit_tangent[0, :]

        # set vertices
        self.set_verts(
            anchor=Vert(
                x=A[0][0] * self.anchor.x + A[0][1] * self.anchor.y,
                y=A[1][0] * self.anchor.x + A[1][1] * self.anchor.y,
            ),
            endpnt=Vert(
                x=A[0][0] * self.endpnt.x + A[0][1] * self.endpnt.y,
                y=A[1][0] * self.endpnt.x + A[1][1] * self.endpnt.y,
            ),
        )

        # determine if the sign of curvature has flipped
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]
        if np.abs(b - c) < TOL and np.abs(a + d) < TOL:
            self.curvature *= -1

    # FUNCTION EVALUATION ####################################################

    def evaluate_function(
        self, fun: Callable, ignore_endpoint: bool = False
    ) -> np.ndarray:
        """Return fun(x) for each sampled point on Edge"""
        if ignore_endpoint:
            k = 1
        else:
            k = 0
        y = np.zeros((self.num_pts - k,))
        for i in range(self.num_pts - k):
            y[i] = fun(self.x[:, i])
        return y

    def multiply_by_dx_norm(
        self, vals: np.ndarray, ignore_endpoint: bool = True
    ) -> np.ndarray:
        """
        Returns f multiplied against the norm of the derivative of
        the curve parameterization
        """
        msg = "vals must be same length as boundary"
        if ignore_endpoint:
            if len(vals) != self.num_pts - 1:
                raise SizeMismatchError(msg)
            return vals * self.dx_norm[:-1]
        if len(vals) != self.num_pts:
            raise SizeMismatchError(msg)
        return vals * self.dx_norm

    def dot_with_tangent(
        self, comp1: np.ndarray, comp2: np.ndarray, ignore_endpoint: bool = True
    ) -> np.ndarray:
        """Returns the dot product (comp1, comp2) * unit_tangent"""
        if ignore_endpoint:
            k = 1
        else:
            k = 0
        if len(comp1) != self.num_pts - k or len(comp2) != self.num_pts - k:
            raise SizeMismatchError("vals must be same length as boundary")
        return (
            comp1 * self.unit_tangent[0, :-k]
            + comp2 * self.unit_tangent[1, :-k]
        )

    def dot_with_normal(
        self, comp1: np.ndarray, comp2: np.ndarray, ignore_endpoint: bool = True
    ) -> np.ndarray:
        """Returns the dot product (comp1, comp2) * unit_normal"""
        if ignore_endpoint:
            k = 1
        else:
            k = 0
        if len(comp1) != self.num_pts - k or len(comp2) != self.num_pts - k:
            raise SizeMismatchError("vals must be same length as boundary")
        return (
            comp1 * self.unit_normal[0, :-k] + comp2 * self.unit_normal[1, :-k]
        )

    # INTEGRATION ############################################################

    def integrate_over_edge(
        self, vals: np.ndarray, ignore_endpoint: bool = False
    ) -> float:
        """Integrate vals * dx_norm over the Edge via trapezoidal rule"""
        vals_dx_norm = self.multiply_by_dx_norm(vals, ignore_endpoint)
        return self.integrate_over_edge_preweighted(
            vals_dx_norm, ignore_endpoint
        )

    def integrate_over_edge_preweighted(
        self, vals_dx_norm: np.ndarray, ignore_endpoint: bool = False
    ) -> float:
        """Integrate vals_dx_norm over the Edge via trapezoidal rule"""
        h = 2 * np.pi / (self.num_pts - 1)
        if ignore_endpoint:
            # left Riemann sum
            if len(vals_dx_norm) != self.num_pts - 1:
                raise SizeMismatchError("vals must be same length as Edge")
            res = np.sum(h * vals_dx_norm[:-1])
        else:
            # trapezoidal rule
            if len(vals_dx_norm) != self.num_pts:
                raise SizeMismatchError("vals must be same length as Edge")
            res = 0.5 * h * (vals_dx_norm[0] + vals_dx_norm[-1]) + np.sum(
                h * vals_dx_norm[1:-1]
            )
        return res
