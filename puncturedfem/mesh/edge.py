"""
edge.py
=======

Module containing the Edge class, which represents an oriented Edge in the
plane.
"""

from typing import Any, Optional, Callable

import numpy as np

# from .bounding_box import get_bounding_box
# from .mesh_exceptions import (
#     EdgeTransformationError,
#     NotParameterizedError,
#     SizeMismatchError,
# )
from .mesh_exceptions import EmbeddingError, NotParameterizedError
from .quad import Quad, QuadDict

from .vert import Vert
from .edge_param import ParameterizedEdge


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
    # num_pts : int
    #     The number of points sampled on the Edge.
    # interp : int
    #     The interpolation parameter
    # x : np.ndarray
    #     The sampled points on the Edge.
    # unit_tangent : np.ndarray
    #     The unit tangent vector at each sampled point on the Edge.
    # unit_normal : np.ndarray
    #     The unit normal vector at each sampled point on the Edge.
    # dx_norm : np.ndarray
    #     The norm of the derivative of the parameterization at each sampled
    #     point on the Edge.
    # curvature : np.ndarray
    #     The signed curvature at each sampled point on the Edge.
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
    param_edge: Optional[ParameterizedEdge]
    param_edge_interp: Optional[ParameterizedEdge]
    interp: int

    # these fields depend on the parameterization
    # num_pts: int
    # interp: int
    # x: np.ndarray
    # unit_tangent: np.ndarray
    # unit_normal: np.ndarray
    # dx_norm: np.ndarray
    # curvature: np.ndarray

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
        self.pos_cell_idx = pos_cell_idx
        self.neg_cell_idx = neg_cell_idx
        self.is_on_mesh_boundary = (
            self.pos_cell_idx < 0 or self.neg_cell_idx < 0
        )

    # PARAMETERIZATION #######################################################

    def parameterize(self, quad_dict: QuadDict) -> None:
        self.is_parameterized = True
        gamma = __import__(
            f"puncturedfem.mesh.edgelib.{self.curve_type}",
            fromlist=f"mesh.edgelib.{self.curve_type}",
        )

        # check for acceptable quadrature type
        if self.quad_type not in ["trap", "kress"]:
            raise ValueError("Quad type not recognized")

        # set quadrature object
        q: Quad = quad_dict[self.quad_type]  # type: ignore

        # set parameterized edge object
        self.param_edge = ParameterizedEdge(
            self.anchor, self.endpnt, q, gamma, **self.curve_opts
        )

        if quad_dict["interp"] == 1:
            self.param_edge_interp = self.param_edge

        elif quad_dict["interp"] > 1:
            # set quadrature object for reduced sampled points
            q_interp: Quad = quad_dict[
                self.quad_type + "_interp" # type: ignore
            ]

            # set parameterized edge object with reduced points
            self.param_edge = ParameterizedEdge(
                self.anchor, self.endpnt, q_interp, gamma, **self.curve_opts
            )

        else:
            raise EmbeddingError("Invalid interpolation parameter")

    def deparameterize(self) -> None:
        """Reset parameterization of the Edge"""
        self.param_edge = None
        self.param_edge_interp = None

    # FUNCTION EVALUATION ####################################################

    def _apply_to_one_parameterized_edge(
        self, method_name: str, interp: int, *args: Any
    ) -> Any:
        """Calls the method on either param_edge or param_edge_interp"""
        if not self.is_parameterized:
            raise NotParameterizedError("calling" + method_name)
        if interp == 1:
            return getattr(self.param_edge, method_name)(args)
        if interp > 1:
            return getattr(self.param_edge_interp, method_name)(args)
        raise ValueError("Invalid interpolation factor")

    def get_sampled_points(
        self, interp: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the sampled points on the Edge"""
        return self._apply_to_one_parameterized_edge(
            "get_sampled_points", interp
        )

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """Return the bounding box of the Edge"""
        return self._apply_to_one_parameterized_edge(
            "get_bounding_box", interp=1
        )

    def evaluate_function(
        self, fun: Callable, ignore_endpoint: bool = False, interp: int = 1
    ) -> np.ndarray:
        """Return fun(x) for each sampled point on Edge"""
        return self._apply_to_one_parameterized_edge(
            "evaluate_function", interp, fun, ignore_endpoint
        )

    def multiply_by_dx_norm(
        self, vals: np.ndarray, ignore_endpoint: bool = True, interp: int = 1
    ) -> np.ndarray:
        """
        Returns f multiplied against the norm of the derivative of
        the curve parameterization
        """
        return self._apply_to_one_parameterized_edge(
            "multiply_by_dx_norm", interp, vals, ignore_endpoint
        )

    def dot_with_tangent(
        self,
        comp1: np.ndarray,
        comp2: np.ndarray,
        ignore_endpoint: bool = True,
        interp: int = 1,
    ) -> np.ndarray:
        """Returns the dot product (comp1, comp2) * unit_tangent"""
        return self._apply_to_one_parameterized_edge(
            "dot_with_tangent", interp, comp1, comp2, ignore_endpoint
        )

    def dot_with_normal(
        self,
        comp1: np.ndarray,
        comp2: np.ndarray,
        ignore_endpoint: bool = True,
        interp: int = 1,
    ) -> np.ndarray:
        """Returns the dot product (comp1, comp2) * unit_normal"""
        return self._apply_to_one_parameterized_edge(
            "dot_with_normal", interp, comp1, comp2, ignore_endpoint
        )

    # INTEGRATION ############################################################

    def integrate_over_edge(
        self, vals: np.ndarray, ignore_endpoint: bool = False, interp: int = 1
    ) -> float:
        """Integrate vals * dx_norm over the Edge via trapezoidal rule"""
        return self._apply_to_one_parameterized_edge(
            "integrate_over_edge", interp, vals, ignore_endpoint
        )

    def integrate_over_edge_preweighted(
        self,
        vals_dx_norm: np.ndarray,
        ignore_endpoint: bool = False,
        interp: int = 1,
    ) -> float:
        """Integrate vals_dx_norm over the Edge via trapezoidal rule"""
        return self._apply_to_one_parameterized_edge(
            "integrate_over_edge_preweighted",
            interp,
            vals_dx_norm,
            ignore_endpoint,
        )

    # TRANSFORMATIONS ########################################################

    def _apply_transformation(self, method_name: str, *args: Any) -> None:
        """Calls the method on both param_edge or param_edge_interp"""
        if not self.is_parameterized:
            raise NotParameterizedError("calling" + method_name)
        if self.param_edge:
            getattr(self.param_edge, method_name)(args)
            self.set_verts(
                anchor=self.param_edge.anchor, endpnt=self.param_edge.endpnt
            )
        if self.param_edge_interp:
            getattr(self.param_edge_interp, method_name)(args)

    def reverse_orientation(self) -> None:
        """
        Reverse the orientation of this Edge using the reparameterization
        x(2 pi - t). The chain rule flips the sign of some derivative-based
        quantities.
        """
        self._apply_transformation("reverse_orientation")

    def join_points(self, a: Vert, b: Vert) -> None:
        """Join the points a to b with this Edge."""
        self._apply_transformation("join_points", a, b)

    def translate(self, z: Vert) -> None:
        """Translate by a vector z"""
        self._apply_transformation("translate", z)

    def dilate(self, alpha: float) -> None:
        """Dilate by a scalar alpha"""
        self._apply_transformation("dilate", alpha)

    def rotate(self, theta: float) -> None:
        """Rotate counterclockwise by theta (degrees)"""
        self._apply_transformation("rotate", theta)

    def reflect_across_x_axis(self) -> None:
        """Reflect across the horizontal axis"""
        self._apply_transformation("reflect_across_x_axis")

    def reflect_across_y_axis(self) -> None:
        """Reflect across the vertical axis"""
        self._apply_transformation("reflect_across_y_axis")

    def apply_orthogonal_transformation(self, A: np.ndarray) -> None:
        """
        Transforms 2-dimensional space with the linear map
                x mapsto A * x
        where A is a 2 by 2 orthogonal matrix, i.e. A^T * A = I

        It is important that A is orthogonal, since the first derivative norm
        as well as the curvature are invariant under such a transformation.
        """
        self._apply_transformation("apply_orthogonal_transformation", A)
