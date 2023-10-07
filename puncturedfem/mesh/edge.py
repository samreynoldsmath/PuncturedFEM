"""
Edge.py
=======

Module containing the Edge class, which represents an oriented Edge in the
plane.
"""

from typing import Any, Callable

import numpy as np

from .bounding_box import get_bounding_box
from .mesh_exceptions import (
    EdgeTransformationError,
    NotParameterizedError,
    SizeMismatchError,
)
from .quad import Quad
from .vert import Vert


class Edge:
    """
    Oriented joining two Vertices of a planar mesh. This class contains both
    the parameterization of the Edge as well as mesh topology information.

    The orientation of the Edge is from the anchor vertex to the endpnt vertex.
    The positive MeshCell is the MeshCell such that the Edge is oriented
    counterclockwise on the boundary of the MeshCell if the Edge lies on the
    outer boundary of the MeshCell. If the Edge lies on the inner boundary of
    the MeshCell, then the Edge is oriented clockwise on the boundary of the
    positive MeshCell. The negative MeshCell is the MeshCell such that the
    boundary of the negative MeshCell intersects the boundary of the positive
    MeshCell exactly on this Edge.

    Usage
    -----
    See examples/ex0-mesh-building.ipynb for examples of how to use this class.

    Attributes
    ----------
    anchor : Vert
        The vertex at the start of the Edge.
    endpnt : Vert
        The vertex at the end of the Edge.
    pos_cell_idx : int
        The index of the positively oriented MeshCell.
    neg_cell_idx : int
        The index of the negatively oriented MeshCell.
    curve_type : str
        The type of curve used to parameterize the Edge.
    curve_opts : dict
        The options for the curve parameterization.
    quad_type : str
        The type of Quadrature used to parameterize the Edge.
    idx : Any
        The global index of the Edge as it appears in the mesh.
    is_on_mesh_boundary : bool
        True if the Edge is on the mesh boundary.
    is_loop : bool
        True if the Edge is a loop.
    is_parameterized : bool
        True if the Edge is parameterized.
    num_pts : int
        The number of points sampled on the Edge.
    x : np.ndarray
        The sampled points on the Edge.
    unit_tangent : np.ndarray
        The unit tangent vector at each sampled point on the Edge.
    unit_normal : np.ndarray
        The unit normal vector at each sampled point on the Edge.
    dx_norm : np.ndarray
        The norm of the derivative of the parameterization at each sampled
        point on the Edge.
    curvature : np.ndarray
        The signed curvature at each sampled point on the Edge.
    """

    anchor: Vert
    endpnt: Vert
    pos_cell_idx: int
    neg_cell_idx: int
    curve_type: str
    curve_opts: dict
    quad_type: str
    idx: Any
    is_on_mesh_boundary: bool
    is_loop: bool
    is_parameterized: bool
    num_pts: int
    x: np.ndarray
    unit_tangent: np.ndarray
    unit_normal: np.ndarray
    dx_norm: np.ndarray
    curvature: np.ndarray

    def __init__(
        self,
        anchor: Vert,
        endpnt: Vert,
        pos_cell_idx: int = -1,
        neg_cell_idx: int = -1,
        curve_type: str = "line",
        quad_type: str = "kress",
        idx: Any = None,
        **curve_opts: Any,
    ) -> None:
        """
        Constructor for the Edge class.

        Parameters
        ----------
        anchor : Vert
            The vertex at the start of the Edge.
        endpnt : Vert
            The vertex at the end of the Edge.
        pos_cell_idx : int, optional
            The index of the positively oriented MeshCell. Default is -1.
        neg_cell_idx : int, optional
            The index of the negatively oriented MeshCell. Default is -1.
        curve_type : str, optional
            The type of curve used to parameterize the Edge. Default is "line".
        quad_type : str, optional
            The type of Quadrature used to parameterize the Edge. Default is
            "kress".
        idx : Any, optional
            The index of the Edge as it appears in the mesh. Default is None.
        """
        self.curve_type = curve_type
        self.quad_type = quad_type
        self.curve_opts = curve_opts
        self.set_idx(idx)
        self.set_verts(anchor, endpnt)
        self.set_cells(pos_cell_idx, neg_cell_idx)
        self.is_parameterized = False

    def __str__(self) -> str:
        """Return a string representation of the Edge"""
        msg = ""
        msg += f"idx:         {self.idx}\n"
        msg += f"curve_type: {self.curve_type}\n"
        msg += f"quad_type:  {self.quad_type}\n"
        msg += f"num_pts:    {self.num_pts}\n"
        return msg

    # MESH TOPOLOGY ##########################################################

    def set_idx(self, idx: Any) -> None:
        """Set the id of the Edge"""
        if idx is None:
            return
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer")
        if idx < 0:
            raise ValueError("idx must be nonnegative")
        self.idx = idx

    def set_verts(self, anchor: Vert, endpnt: Vert) -> None:
        """Set the anchor and endpnt Vertices of the Edge"""
        self.anchor = anchor
        self.endpnt = endpnt
        self.is_loop = self.anchor == self.endpnt

    def set_cells(self, pos_cell_idx: int, neg_cell_idx: int) -> None:
        """
        Set the positively and negatively oriented MeshCells of the Edge.
        """

        # check that MeshCells are distinct
        # TODO this warning should only happen in PlanarMesh
        # if pos_cell_idx < 0 and neg_cell_idx < 0:
        #     raise ValueError(
        #         'Edge must be boundary of at least one MeshCell'
        #     )

        self.pos_cell_idx = pos_cell_idx
        self.neg_cell_idx = neg_cell_idx
        self.is_on_mesh_boundary = (
            self.pos_cell_idx < 0 or self.neg_cell_idx < 0
        )

    # PARAMETERIZATION #######################################################

    def parameterize(self, quad_dict: dict[str, Quad]) -> None:
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

        q = quad_dict[self.quad_type]

        # 2 * n + 1 points sampled per Edge
        self.num_pts = 2 * q.n + 1

        # retrieve function handles of parameterization and derivatives
        gamma = __import__(
            f"puncturedfem.mesh.edgelib.{self.curve_type}",
            fromlist=f"mesh.edgelib.{self.curve_type}",
        )

        # points on the boundary
        self.x = gamma.X(q.t, **self.curve_opts)

        # unweighted square norm of derivative
        dx = gamma.DX(q.t, **self.curve_opts)
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
        ddx = gamma.DDX(q.t, **self.curve_opts)
        self.curvature = (
            ddx[0, :] * self.unit_normal[0, :]
            + ddx[1, :] * self.unit_normal[1, :]
        ) / dx2

        # toggle parameterization flag
        self.is_parameterized = True

        # set endpoints
        if self.is_loop:
            self.translate(a=self.anchor)
        else:
            self.join_points(a=self.anchor, b=self.endpnt)

        # store parameterization type
        self.quad_type = q.type

    def deparameterize(self) -> None:
        """Reset the parameterization of the Edge"""
        self.num_pts = -1
        self.x = np.zeros((0,))
        self.unit_tangent = np.zeros((0,))
        self.unit_normal = np.zeros((0,))
        self.dx_norm = np.zeros((0,))
        self.curvature = np.zeros((0,))
        self.is_parameterized = False

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

        # check if Edge is parameterized
        if not self.is_parameterized:
            raise NotParameterizedError("reversing orientation")

        # vector quantities
        self.x = np.fliplr(self.x)
        self.unit_tangent = -np.fliplr(self.unit_tangent)
        self.unit_normal = -np.fliplr(self.unit_normal)

        # scalar quantities
        self.dx_norm = np.flip(self.dx_norm)
        self.curvature = -np.flip(self.curvature)

    def join_points(self, a: Vert, b: Vert) -> None:
        """Join the points a to b with this Edge."""

        # check if Edge is parameterized
        if not self.is_parameterized:
            raise NotParameterizedError("joining points")

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

        # set Vertices
        self.set_verts(a, b)

    def translate(self, a: Vert) -> None:
        """Translate by a vector a"""

        # check if Edge is parameterized
        if not self.is_parameterized:
            raise NotParameterizedError("translating")

        self.x[0, :] += a.x
        self.x[1, :] += a.y

    def dilate(self, alpha: float) -> None:
        """Dilate by a scalar alpha"""

        # check if Edge is parameterized
        if not self.is_parameterized:
            raise NotParameterizedError("dilating")

        # tolerance for floating point comparisons
        TOL = 1e-12

        if np.abs(alpha) < TOL:
            raise EdgeTransformationError(
                "Dilation factor alpha must be nonzero"
            )

        self.x *= alpha
        self.dx_norm *= np.abs(alpha)
        self.curvature *= 1 / alpha

    def rotate(self, theta: float) -> None:
        """Rotate counterclockwise by theta (degrees)"""

        # check if Edge is parameterized
        if not self.is_parameterized:
            raise NotParameterizedError("rotating")

        if theta % 360 == 0:
            return

        c = np.cos(theta * np.pi / 180)
        s = np.sin(theta * np.pi / 180)
        R = np.array([[c, -s], [s, c]])

        self.apply_orthogonal_transformation(R)

    def reflect_across_x_axis(self) -> None:
        """Reflect across the horizontal axis"""

        # check if Edge is parameterized
        if not self.is_parameterized:
            raise NotParameterizedError("reflecting across x axis")

        A = np.array([[1, 0], [0, -1]])
        self.apply_orthogonal_transformation(A)

    def reflect_across_y_axis(self) -> None:
        """Reflect across the vertical axis"""
        if not self.is_parameterized:
            raise NotParameterizedError("reflecting across y axis")

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

        # check if Edge is parameterized
        if not self.is_parameterized:
            raise NotParameterizedError("applying orthogonal transformation")

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
        if not self.is_parameterized:
            raise NotParameterizedError("evaluating function")
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
        if not self.is_parameterized:
            raise NotParameterizedError("multiplying by dx_norm")
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
        if not self.is_parameterized:
            raise NotParameterizedError("dotting with tangent")
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
        if not self.is_parameterized:
            raise NotParameterizedError("dotting with normal")
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
        if not self.is_parameterized:
            raise NotParameterizedError("integrating over Edge")
        vals_dx_norm = self.multiply_by_dx_norm(vals, ignore_endpoint)
        return self.integrate_over_edge_preweighted(
            vals_dx_norm, ignore_endpoint
        )

    def integrate_over_edge_preweighted(
        self, vals_dx_norm: np.ndarray, ignore_endpoint: bool = False
    ) -> float:
        """Integrate vals_dx_norm over the Edge via trapezoidal rule"""
        if not self.is_parameterized:
            raise NotParameterizedError("integrating over Edge")
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
