"""
Edge.py
=======

Module containing the Edge class, which represents an oriented Edge in the
plane.
"""

from os import path
from typing import Any

import numpy as np

from ..util.types import Func_R2_R
from .bounding_box import get_bounding_box
from .mesh_exceptions import (
    EdgeTransformationError,
    NotParameterizedError,
    SizeMismatchError,
)
from .quad import QuadDict
from .vert import Vert

# tolerance for floating point comparisons
TOL = 1e-12


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
    gamma: Any
    t_bounds: tuple[float, float]
    diary: list[tuple[str, tuple]]

    def __init__(
        self,
        anchor: Vert,
        endpnt: Vert,
        pos_cell_idx: int = -1,
        neg_cell_idx: int = -1,
        curve_type: str = "line",
        quad_type: str = "kress",
        idx: Any = None,
        t_bounds: tuple[float, float] = (0.0, 2 * np.pi),
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
        self.diary = []
        self.set_idx(idx)
        self.set_t_bounds(t_bounds)
        self.set_verts(anchor, endpnt)
        self.set_cells(pos_cell_idx, neg_cell_idx)
        self.is_parameterized = False
        self.num_pts = -1

    def __str__(self) -> str:
        """Return a string representation of the Edge"""
        msg = ""
        if hasattr(self, "idx"):
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
        if self.is_loop:
            self.diary = [("translate", (anchor,))]
        else:
            self.diary = [("join_points", (anchor, endpnt))]

    def set_cells(self, pos_cell_idx: int, neg_cell_idx: int) -> None:
        """
        Set the positively and negatively oriented MeshCells of the Edge.
        """
        self.pos_cell_idx = pos_cell_idx
        self.neg_cell_idx = neg_cell_idx
        self.is_on_mesh_boundary = (
            self.pos_cell_idx < 0 or self.neg_cell_idx < 0
        )

    # PARAMETERIZATION #######################################################

    def set_curve_type(self, curve_type: str) -> None:
        """
        Set the curve_type string.
        """
        if not isinstance(curve_type, str):
            raise ValueError("curve type must be a str")
        dirname = path.dirname(__file__)
        filename = path.join(dirname, "edgelib/" + self.curve_type + ".py")
        if not path.exists(filename):
            raise FileExistsError(filename + "does not exist")
        self.curve_type = curve_type

    def set_t_bounds(self, t_bounds: tuple[float, float]) -> None:
        """
        Set the bounds on t, where x(t) parameterizes the curve.
        """
        msg = "t_bounds must be a pair of floats with t_bounds[0] < t_bounds[1]"
        a, b = t_bounds
        if not (isinstance(a, (float, int)) and isinstance(b, (float, int))):
            raise ValueError(msg, t_bounds)
        if a >= b:
            raise ValueError(msg, t_bounds)
        self.t_bounds = t_bounds

    def run_transform_diary(self) -> None:
        """
        Execute all the transformations in the diary.
        """
        for method_name, args in self.diary:
            method = getattr(self, method_name)
            method(*args, write_diary=False)

    def get_parameterization_module(self) -> Any:
        """
        Returns the module with the functions X, DX, DDX
        """
        return __import__(
            f"puncturedfem.mesh.edgelib.{self.curve_type}",
            fromlist=f"mesh.edgelib.{self.curve_type}",
        )

    def parameterize(self, quad_dict: QuadDict) -> None:
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

        # check for acceptable quadrature type
        if self.quad_type not in ["trap", "kress"]:
            raise ValueError("Quad type not recognized")

        # set quadrature object
        q = quad_dict[self.quad_type]  # type: ignore

        # 2 * n + 1 points sampled per Edge
        self.num_pts = 2 * q.n + 1

        # rescale sampling interval
        a, b = self.t_bounds
        dt = 0.5 * (b - a) / np.pi
        t = dt * q.t + a

        # load module with X, DX, DDX functions
        gamma = self.get_parameterization_module()

        # points on the boundary
        self.x = gamma.X(t, **self.curve_opts)

        # unweighted square norm of derivative
        dx = dt * gamma.DX(t, **self.curve_opts)
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
        ddx = dt**2 * gamma.DDX(t, **self.curve_opts)
        self.curvature = (
            ddx[0, :] * self.unit_normal[0, :]
            + ddx[1, :] * self.unit_normal[1, :]
        ) / dx2

        # toggle parameterization flag
        self.is_parameterized = True

        # apply transformations
        self.run_transform_diary()

        # if self.is_loop:
        #     self.translate(self.anchor)
        # else:
        #     self.join_points(self.anchor, self.endpnt)

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

    def get_sampled_points(
        self, ignore_endpoint: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the sampled points on the Edge"""
        if not self.is_parameterized:
            raise NotParameterizedError("getting sampled points")
        if ignore_endpoint:
            return self.x[0, :-1], self.x[1, :-1]
        return self.x[0, :], self.x[1, :]

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """Return the bounding box of the Edge"""
        if not self.is_parameterized:
            raise NotParameterizedError("getting bounding box")
        return get_bounding_box(x=self.x[0, :], y=self.x[1, :])

    # TRANSFORMATION CONVENIENCE METHODS ######################################

    def join_points(self, a: Vert, b: Vert, write_diary: bool = True) -> None:
        """Join the points a to b with this Edge."""

        # check that specified endpoints are distinct
        ab_norm = np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
        if ab_norm < TOL:
            raise EdgeTransformationError("a and b must be distinct points")

        # record in transformation diary
        if write_diary:
            self.diary.append(("join_points", (a, b)))

        # apply to sampled points
        if self.is_parameterized:
            # check that endpoints of sampled points are distinct
            x = self.x[:, 0]
            y = self.x[:, self.num_pts - 1]
            xy_norm = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
            if xy_norm < TOL:
                raise EdgeTransformationError(
                    "Edge must have distinct endpoints"
                )

            # anchor starting point to origin
            self.translate(Vert(x=-x[0], y=-x[1]), write_diary=False)

            # rotate
            theta = -np.arctan2(y[1] - x[1], y[0] - x[0])
            theta += np.arctan2(b.y - a.y, b.x - a.x)
            theta *= 180 / np.pi
            self.rotate(theta, write_diary=False)

            # rescale
            alpha = ab_norm / xy_norm
            self.dilate(alpha, write_diary=False)

            # anchor at point a
            self.translate(a, write_diary=False)

            # set vertices
            self.set_verts(a, b)

    def rotate(self, theta: float, write_diary: bool = True) -> None:
        """Rotate counterclockwise by theta (degrees)"""

        # record in transformation diary
        if write_diary:
            self.diary.append(("rotate", (theta,)))

        # apply to sampled points
        if self.is_parameterized:
            c = np.cos(theta * np.pi / 180)
            s = np.sin(theta * np.pi / 180)
            R = np.array([[c, -s], [s, c]])
            self.apply_orthogonal_transformation(R, write_diary=False)

    def reflect_across_x_axis(self, write_diary: bool = True) -> None:
        """Reflect across the horizontal axis"""

        # record in transformation diary
        if write_diary:
            self.diary.append(("reflect_across_x_axis", ()))

        # apply to sampled points
        if self.is_parameterized:
            A = np.array([[1, 0], [0, -1]])
            self.apply_orthogonal_transformation(A, write_diary=False)

    def reflect_across_y_axis(self, write_diary: bool = True) -> None:
        """Reflect across the vertical axis"""

        # record in transformation diary
        if write_diary:
            self.diary.append(("reflect_across_y_axis", ()))

        # apply to sampled points
        if self.is_parameterized:
            A = np.array([[-1, 0], [0, 1]])
            self.apply_orthogonal_transformation(A, write_diary=False)

    # TRANSFORMATION BASE METHODS #############################################

    def reverse_orientation(self) -> None:
        """
        Reverse the orientation of this Edge using the reparameterization
        x(2 pi - t). The chain rule flips the sign of some derivative-based
        quantities.
        """

        # NOTE: DO NOT record in transformation diary

        # apply to sampled points
        if self.is_parameterized:
            # vector quantities
            self.x = np.fliplr(self.x)
            self.unit_tangent = -np.fliplr(self.unit_tangent)
            self.unit_normal = -np.fliplr(self.unit_normal)

            # scalar quantities
            self.dx_norm = np.flip(self.dx_norm)
            self.curvature = -np.flip(self.curvature)

    def translate(self, a: Vert, write_diary: bool = True) -> None:
        """Translate by a vector a"""

        # record in transformation diary
        if write_diary:
            self.diary.append(("translate", (a,)))

        # apply to sampled points
        if self.is_parameterized:
            self.x[0, :] += a.x
            self.x[1, :] += a.y

    def dilate(self, alpha: float, write_diary: bool = True) -> None:
        """Dilate by a scalar alpha"""

        # check that alpha is nonzero
        if np.abs(alpha) < TOL:
            raise EdgeTransformationError(
                "Dilation factor alpha must be nonzero"
            )

        # record in transformation diary
        if write_diary:
            self.diary.append(("dilate", (alpha,)))

        # apply to sampled points
        if self.is_parameterized:
            self.x *= alpha
            self.dx_norm *= np.abs(alpha)
            self.curvature *= 1 / alpha

    def apply_orthogonal_transformation(
        self, A: np.ndarray, write_diary: bool = True
    ) -> None:
        """
        Transforms 2-dimensional space with the linear map
                x mapsto A * x
        where A is a 2 by 2 orthogonal matrix, i.e. A^T * A = I

        It is important that A is orthogonal, since the first derivative norm
        as well as the curvature are invariant under such a transformation.
        """

        # safety checks
        msg = "A must be a 2 by 2 orthogonal matrix"
        if np.shape(A) != (2, 2):
            raise EdgeTransformationError(msg)
        if np.linalg.norm(np.transpose(A) @ A - np.eye(2)) > TOL:
            raise EdgeTransformationError(msg)

        # record in transformation diary
        if write_diary:
            self.diary.append(("apply_orthogonal_transformation", (A,)))

        # apply to sampled points
        if self.is_parameterized:
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
        self, fun: Func_R2_R, ignore_endpoint: bool = True
    ) -> np.ndarray:
        """Return fun(x1, x2) for each sampled point on Edge"""
        if not self.is_parameterized:
            raise NotParameterizedError("evaluating function")
        if ignore_endpoint:
            k = 1
        else:
            k = 0
        y = np.zeros((self.num_pts - k,))
        for i in range(self.num_pts - k):
            y[i] = fun(self.x[0, i], self.x[1, i])
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
