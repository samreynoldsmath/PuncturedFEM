"""
ClosedContour.py
=================

Module containing the ClosedContour class, which represents a closed contour
in the plane.
"""

from typing import Optional

import numpy as np
from matplotlib import path

from ..util.types import Func_R2_R
from .bounding_box import get_bounding_box
from .edge import Edge
from .mesh_exceptions import (
    InteriorPointError,
    NotParameterizedError,
    SizeMismatchError,
)
from .quad import QuadDict
from .vert import Vert


class ClosedContour:
    """
    List of edges forming a closed contour, assumed to be simple
    with edges listed successively.
    """

    edges: list[Edge]
    num_edges: int
    edge_orient: list[int]
    interior_point: Vert
    num_pts: int
    vert_idx: list[int]
    local_vert_idx: list[int]
    closest_vert_idx: np.ndarray

    def __init__(
        self,
        cell_id: int,
        edges: Optional[list[Edge]] = None,
        edge_orients: Optional[list[int]] = None,
    ) -> None:
        """
        Constructor for ClosedContour class.

        Parameters
        ----------
        cell_id: int
            The MeshCell id of the contour. (This is the MeshCell id of the
            MeshCell that the contour is part of the boundary of.)
        """
        self.set_mesh_id(cell_id)
        self.num_edges = 0
        self.vert_idx = []
        self.edges: list[Edge] = []
        self.edge_orients: list[int] = []
        if edges is None:
            edges = []
        if edge_orients is None:
            edge_orients = []
        self.add_edges(edges, edge_orients)

    def set_mesh_id(self, cell_id: int) -> None:
        """Set the MeshCell id of the contour"""
        if not isinstance(cell_id, int):
            raise TypeError(
                f"cell_id = {cell_id} invalid, must be a positive integer"
            )
        if cell_id < 0:
            raise ValueError(
                f"cell_id = {cell_id} invalid, must be a positive integer"
            )
        self.mesh_id = cell_id

    # EDGE MANAGEMENT ########################################################
    def add_edge(self, e: Edge, edge_orient: int) -> None:
        """Add Edge to contour"""
        if edge_orient not in (+1, -1):
            raise ValueError("Orientation must be +1 or -1")
        if e in self.edges:
            return
        self.edges.append(e)
        self.edge_orients.append(edge_orient)
        self.num_edges += 1

    def add_edges(self, edges: list[Edge], edge_orients: list[int]) -> None:
        """Add edges to contour"""
        if len(edges) != len(edge_orients):
            raise ValueError("Must provide orientation for each Edge")
        for e, o in zip(edges, edge_orients):
            self.add_edge(e, o)

    def is_closed(self) -> bool:
        """Returns true if the contour is closed"""
        raise NotImplementedError()

    # PARAMETERIZATION #######################################################
    def is_parameterized(self) -> bool:
        """Returns true if all edges are parameterized"""
        return all(e.is_parameterized for e in self.edges)

    def parameterize(self, quad_dict: QuadDict) -> None:
        """Parameterize each Edge"""
        # TODO: eliminate redundant calls to parameterize
        for i in range(self.num_edges):
            self.edges[i].parameterize(quad_dict)
            if self.edge_orients[i] == -1:
                self.edges[i].reverse_orientation()
        self.find_num_pts()
        self.find_local_vert_idx()
        self.find_closest_local_vertex_index()
        self.find_interior_point()

    def deparameterize(self) -> None:
        """Deparameterize each Edge"""
        for e in self.edges:
            e.deparameterize()
        self.num_pts = 0
        self.closest_vert_idx = np.zeros((0,))

    def find_num_pts(self) -> None:
        """Record the total number of sampled points on the boundary"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding num_pts")
        self.num_pts = 0
        for e in self.edges:
            self.num_pts += e.num_pts - 1

    def find_local_vert_idx(self) -> None:
        """Get the index of the starting point of each Edge"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding vert_idx")
        self.vert_idx = [0]
        for e in self.edges:
            self.vert_idx.append(self.vert_idx[-1] + e.num_pts - 1)

    def find_closest_local_vertex_index(self) -> None:
        """Find the index of the closest vertex for each sampled point"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding closest_vert_idx")

        # get midpoint indices
        mid_idx = np.zeros((self.num_edges,), dtype=int)
        for i in range(self.num_edges):
            n = self.edges[i].num_pts // 2  # 2n points per Edge
            mid_idx[i] = self.vert_idx[i] + n

        # on first half of an Edge, the closest vertex is the starting
        # point on that Edge; on the second half of an Edge, the closest vertex
        # is the starting point of the next Edge
        self.closest_vert_idx = np.zeros((self.num_pts,), dtype=int)
        for i in range(self.num_edges):
            self.closest_vert_idx[self.vert_idx[i] : mid_idx[i]] = (
                self.vert_idx[i]
            )
            self.closest_vert_idx[mid_idx[i] : self.vert_idx[i + 1]] = (
                self.vert_idx[i + 1] % self.num_pts
            )

    # INTERIOR POINTS ########################################################
    def get_distance_to_boundary(self, x: float, y: float) -> float:
        """Minimum distance from (x,y) to a point on the boundary"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding distance to boundary")
        dist = np.inf
        for e in self.edges:
            dist2e = min((e.x[0, :] - x) ** 2 + (e.x[1, :] - y) ** 2)
            dist = min([dist, dist2e])
        return np.sqrt(dist)

    def is_in_interior_contour(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Returns a boolean array indicating whether each point (x[i], y[i]) is
        inside the contour.
        """
        if x.shape != y.shape:
            raise SizeMismatchError("x and y must have same size")

        is_inside = np.zeros(x.shape, dtype=bool)
        x1, x2 = self.get_sampled_points()
        p = path.Path(np.array([x1, x2]).transpose())

        if len(x.shape) == 1:
            M = x.shape[0]
            for i in range(M):
                pt = (x[i], y[i])
                is_inside[i] = p.contains_point(pt)
        elif len(x.shape) == 2:
            M, N = x.shape
            for i in range(M):
                for j in range(N):
                    pt = (x[i, j], y[i, j])
                    is_inside[i, j] = p.contains_point(pt)

        return is_inside

    def find_interior_point(self) -> None:
        """Finds an interior point."""

        # NOTE: Uses a brute force search. There is likely a more efficient way.

        if not self.is_parameterized():
            raise NotParameterizedError("finding interior point")

        # find region of interest
        x, y = self.get_sampled_points()
        xmin, xmax, ymin, ymax = get_bounding_box(x, y)

        # set minimum desired distance to the boundary
        TOL = 0.01 * min([xmax - xmin, ymax - ymin])

        # search from M by N rectangular grid points
        M = 9
        N = 9

        # initialize distance to boundary
        d = 0.0

        while d < TOL:
            # set up grid
            x_coord = np.linspace(xmin, xmax, M)
            y_coord = np.linspace(ymin, ymax, N)
            x, y = np.meshgrid(x_coord, y_coord)

            # determine which points are in the interior
            is_inside = self.is_in_interior_contour(x, y)

            # for each interior point in grid, compute distance to the boundary
            dist = np.zeros(np.shape(x))
            for i in range(M):
                for j in range(N):
                    if is_inside[i, j]:
                        dist[i, j] = self.get_distance_to_boundary(
                            x[i, j], y[i, j]
                        )

            # pick a point farthest from the boundary
            k = np.argmax(dist, keepdims=True)
            ii = k[0][0] // M
            jj = k[0][0] % M
            d = dist[ii, jj]

            # if the best candidate is too close to the boundary,
            # refine grid and search again
            M = 4 * (M // 2) + 1
            N = 4 * (N // 2) + 1

            if M * N > 1_000_000:
                raise InteriorPointError("Unable to locate an interior point")

            self.interior_point = Vert(x=x[ii, jj], y=y[ii, jj])

    # FUNCTION EVALUATION ####################################################
    def evaluate_function_on_contour(self, fun: Func_R2_R) -> np.ndarray:
        """Return fun(x) for each sampled point on contour"""
        if not self.is_parameterized():
            raise NotParameterizedError("evaluating function on contour")
        y = np.zeros((self.num_pts,))
        for j in range(self.num_edges):
            y[self.vert_idx[j] : self.vert_idx[j + 1]] = self.edges[
                j
            ].evaluate_function(fun, ignore_endpoint=True)
        return y

    def get_sampled_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the x1 and x2 coordinates of the boundary points"""
        if not self.is_parameterized():
            raise NotParameterizedError("getting boundary points")
        x1 = self.evaluate_function_on_contour(lambda x1, x2: x1)
        x2 = self.evaluate_function_on_contour(lambda x1, x2: x2)
        return x1, x2

    def dot_with_tangent(
        self, comp1: np.ndarray, comp2: np.ndarray
    ) -> np.ndarray:
        """Returns the dot product (comp1, comp2) * unit_tangent"""
        if not self.is_parameterized():
            raise NotParameterizedError("dotting with tangent")
        res = np.zeros((self.num_pts,))
        for i in range(self.num_edges):
            j = self.vert_idx[i]
            jp1 = self.vert_idx[i + 1]
            res[j:jp1] = self.edges[i].dot_with_tangent(
                comp1[j:jp1], comp2[j:jp1]
            )
        return res

    def dot_with_normal(
        self, comp1: np.ndarray, comp2: np.ndarray
    ) -> np.ndarray:
        """Returns the dot product (comp1, comp2) * unit_normal"""
        if not self.is_parameterized():
            raise NotParameterizedError("dotting with normal")
        res = np.zeros((self.num_pts,))
        for i in range(self.num_edges):
            j = self.vert_idx[i]
            jp1 = self.vert_idx[i + 1]
            res[j:jp1] = self.edges[i].dot_with_normal(
                comp1[j:jp1], comp2[j:jp1]
            )
        return res

    def multiply_by_dx_norm(self, vals: np.ndarray) -> np.ndarray:
        """
        Returns f multiplied against the norm of the derivative of
        the curve parameterization
        """
        if not self.is_parameterized():
            raise NotParameterizedError("multiplying by dx_norm")
        if len(vals) != self.num_pts:
            raise SizeMismatchError("vals must be same length as boundary")
        vals_dx_norm = np.zeros((self.num_pts,))
        for i in range(self.num_edges):
            j = self.vert_idx[i]
            jp1 = self.vert_idx[i + 1]
            vals_dx_norm[j:jp1] = self.edges[i].multiply_by_dx_norm(vals[j:jp1])
        return vals_dx_norm

    # INTEGRATION ############################################################
    def integrate_over_closed_contour(self, vals: np.ndarray) -> float:
        """Contour integral of vals"""
        if not self.is_parameterized():
            raise NotParameterizedError("integrating over boundary")
        vals_dx_norm = self.multiply_by_dx_norm(vals)
        return self.integrate_over_closed_contour_preweighted(vals_dx_norm)

    def integrate_over_closed_contour_preweighted(
        self, vals_dx_norm: np.ndarray
    ) -> float:
        """Contour integral of vals_dx_norm"""

        # check inputs
        if not self.is_parameterized():
            raise NotParameterizedError("integrating over boundary")
        if len(np.shape(vals_dx_norm)) != 1:
            raise SizeMismatchError("vals_dx_norm must be a vector")
        if len(vals_dx_norm) != self.num_pts:
            raise SizeMismatchError("vals must be same length as boundary")

        # NOTE: numpy.sum() is more stable, but this uses more memory
        y = np.zeros((self.num_pts,))
        for i in range(self.num_edges):
            h = 2 * np.pi / (self.edges[i].num_pts - 1)
            y[self.vert_idx[i] : self.vert_idx[i + 1]] = (
                h * vals_dx_norm[self.vert_idx[i] : self.vert_idx[i + 1]]
            )
        return float(np.sum(y))
