"""
cell.py
=======

Module containing the cell class, used to represent a mesh cell.
"""

import numpy as np

from ..util.types import Func_R2_R
from .closed_contour import ClosedContour
from .edge import Edge
from .mesh_exceptions import (
    EmbeddingError,
    NotParameterizedError,
    SizeMismatchError,
)
from .quad import QuadDict


class MeshCell:
    """
    Class representing a mesh cell, which may be multiply connected.

    Contains:
        - parameterization of the boundary
        - methods to evaluate and integrate functions on the boundary
        - interior points
        - relative position of the cell in the mesh (topological info)

    Attributes
    ----------
    idx : int
        The cell id as it appears in the mesh.
    components : list[ClosedContour]
        The boundary components of the cell.
    num_holes : int
        The number of holes in the cell.
    num_edges : int
        The number of edges on the cell boundary.
    num_pts : int
        The number of sampled points on the cell boundary.
    component_start_idx : list[int]
        The index of the first sampled point on each boundary component.
    closest_vert_idx : np.ndarray
        The index of the closest vertex in the mesh to each sampled point.
    edge_orients : list[int]
        The orientation of each Edge in the cell (+/- 1).
    int_mesh_size : tuple[int, int]
    """

    idx: int
    components: list[ClosedContour]
    num_holes: int
    num_edges: int
    num_pts: int
    quad_dict: QuadDict
    component_start_idx: list[int]
    closest_vert_idx: np.ndarray
    edge_orients: list[int]
    int_mesh_size: tuple[int, int]
    int_x1: np.ndarray
    int_x2: np.ndarray
    is_inside: np.ndarray

    def __init__(
        self,
        idx: int,
        edges: list[Edge],
        int_mesh_size: tuple[int, int] = (101, 101),
        rtol: float = 0.02,
        atol: float = 0.02,
    ) -> None:
        """
        Constructor for the cell class.

        Parameters
        ----------
        id : int
            The cell id.
        edges : list[Edge]
            The edges in the cell.
        """
        self.set_idx(idx)
        self.find_edge_orientations(edges)
        self.components = []
        self.find_boundary_components(edges)
        self.find_num_edges()
        self.set_interior_mesh_size(*int_mesh_size)
        self.set_interior_point_tolerance(rtol, atol)

    # MESH TOPOLOGY ##########################################################

    def set_idx(self, idx: int) -> None:
        """Set the global cell index"""
        if not isinstance(idx, int):
            raise TypeError(f"idx = {idx} invalid, must be a positive integer")
        if idx < 0:
            raise ValueError(f"idx = {idx} invalid, must be a positive integer")
        self.idx = idx

    def find_edge_orientations(self, edges: list[Edge]) -> None:
        """Find the orientation of each Edge in the cell"""
        self.edge_orients = []
        for e in edges:
            if self.idx == e.pos_cell_idx:
                self.edge_orients.append(+1)
            elif self.idx == e.neg_cell_idx:
                self.edge_orients.append(-1)
            else:
                print(f"cell idx = {self.idx}")
                print(f"Edge pos_cell_idx = {e.pos_cell_idx}")
                print(f"Edge neg_cell_idx = {e.neg_cell_idx}")
                raise EmbeddingError("Undefined Edge orientation")
                # self.edge_orients.append(0)

    # LOCAL EDGE MANAGEMENT ##################################################

    def find_num_edges(self) -> None:
        """Find the number of edges in the cell"""
        self.num_edges = sum(c.num_edges for c in self.components)

    def get_edges(self) -> list[Edge]:
        """Returns a list of all edges in the cell"""
        edges = []
        for c in self.components:
            for e in c.edges:
                edges.append(e)
        return edges

    def get_edge_endpoint_incidence(self, edges: list[Edge]) -> np.ndarray:
        """
        Returns incidence array: for each Edge i, point to an Edge j
        whose starting point is the terminal point of Edge i

                Edge i          vertex     Edge j
                --->--->--->--- o --->--->--->---
        """
        # if not self.is_parameterized():
        #     raise NotParameterizedError('finding Edge endpoint incidence')

        # form distance matrix between endpoints of edges
        num_edges = len(edges)
        distance = np.zeros((num_edges, num_edges))
        for i in range(num_edges):
            if self.edge_orients[i] == +1:
                a = edges[i].anchor
            elif self.edge_orients[i] == -1:
                a = edges[i].endpnt
            else:
                raise EmbeddingError("Edge orientation must be +1 or -1")
            for j in range(num_edges):
                if self.edge_orients[j] == +1:
                    b = edges[j].endpnt
                elif self.edge_orients[j] == -1:
                    b = edges[j].anchor
                else:
                    raise EmbeddingError("Edge orientation must be +1 or -1")
                distance[i, j] = np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

        # mark edges as incident if distance between endpoints is zero
        TOL = 1e-6
        incidence_mat = np.zeros(distance.shape, dtype=int)
        for i in range(num_edges):
            for j in range(num_edges):
                if distance[i, j] < TOL:
                    incidence_mat[i, j] = 1

        # check that each Edge endpoint is incident to exactly one other Edge
        row_sum = np.sum(incidence_mat, axis=0)
        rows_all_sum_to_one = np.linalg.norm(row_sum - 1) < TOL

        col_sum = np.sum(incidence_mat, axis=1)
        cols_all_sum_to_one = np.linalg.norm(col_sum - 1) < TOL

        if not (rows_all_sum_to_one and cols_all_sum_to_one):
            raise EmbeddingError(
                "Edge collection must be a union of "
                + "disjoint simple closed contours"
            )

        # for each Edge, return the index of the Edge following it
        incidence = np.zeros((num_edges,), dtype=int)
        for i in range(num_edges):
            j = 0
            while incidence_mat[i, j] == 0:
                j += 1
            incidence[j] = i

        return incidence

    def find_boundary_components(self, edges: list[Edge]) -> None:
        """Finds the boundary components of the cell"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding closed contours")

        contour_idx = []
        num_edges = len(edges)
        incidence = self.get_edge_endpoint_incidence(edges)

        is_marked_edge = np.zeros((num_edges,), dtype=bool)
        num_marked_edges = 0

        while num_marked_edges < num_edges:
            edges_on_contour = []
            starting_edge = 0

            while is_marked_edge[starting_edge]:
                starting_edge += 1

            edges_on_contour.append(starting_edge)
            is_marked_edge[starting_edge] = True
            next_edge = incidence[starting_edge]

            while next_edge != starting_edge:
                edges_on_contour.append(next_edge)
                is_marked_edge[next_edge] = True
                next_edge = incidence[next_edge]

            num_marked_edges += len(edges_on_contour)

            contour_idx.append(edges_on_contour)

        self.num_holes = -1 + len(contour_idx)

        for c_idx in contour_idx:
            edges_c = [edges[i] for i in c_idx]
            edge_orients_c = [self.edge_orients[i] for i in c_idx]
            self.components.append(
                ClosedContour(
                    cell_id=self.idx, edges=edges_c, edge_orients=edge_orients_c
                )
            )

    def find_hole_interior_points(self) -> None:
        """
        DEPRECATED: Automatically find a point in the interior of each hole.

        Finds a point by creating a rectangular grid of points and
        eliminating those that are not in the interior. Among those
        that are in the interior, a point that lies a maximum distance
        from the boundary is chosen.
        """
        raise NotImplementedError()

    # PARAMETERIZATION #######################################################
    def is_parameterized(self) -> bool:
        """Returns True if the cell is parameterized"""
        return all(c.is_parameterized() for c in self.components)

    def parameterize(self, quad_dict: QuadDict) -> None:
        """Parameterize each Edge"""
        for c in self.components:
            c.parameterize(quad_dict)
        self.find_num_pts()
        self.find_outer_boundary()
        self.find_component_start_idx()
        self.find_closest_vert_idx()
        self.generate_interior_points()
        self.quad_dict = quad_dict

    def deparameterize(self) -> None:
        """Remove parameterization of each Edge"""
        for c in self.components:
            c.deparameterize()
        self.num_pts = 0
        self.component_start_idx = []

    def find_num_pts(self) -> None:
        """Record the total number of sampled points on the boundary"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding num_pts")
        self.num_pts = sum(c.num_pts for c in self.components)

    def find_component_start_idx(self) -> None:
        """Find the index of sampled points corresponding to each component"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding component_start_idx")
        self.component_start_idx = []
        idx = 0
        for c in self.components:
            self.component_start_idx.append(idx)
            idx += c.num_pts
        self.component_start_idx.append(idx)

    def find_outer_boundary(self) -> None:
        """Find the outer boundary of the cell"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding outer boundary")
        # find component that contains all other components
        outer_boundary_idx = 0
        for i in range(self.num_holes + 1):
            for j in range(i + 1, self.num_holes + 1):
                # check if contour j is contained in contour i
                x1, x2 = self.components[j].get_sampled_points()
                is_inside = self.components[i].is_in_interior_contour(x1, x2)
                if all(is_inside):
                    outer_boundary_idx = i
        # swap contour_idx[0] and the outer boundary index
        temp = self.components[0]
        self.components[0] = self.components[outer_boundary_idx]
        self.components[outer_boundary_idx] = temp

    def find_closest_vert_idx(self) -> None:
        """Find the closest vertex in the mesh to each sampled point"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding closest_vert_idx")
        self.closest_vert_idx = np.zeros((self.num_pts,), dtype=int)
        for i in range(self.num_holes + 1):
            j = self.component_start_idx[i]
            jp1 = self.component_start_idx[i + 1]
            self.closest_vert_idx[j:jp1] = self.components[i].closest_vert_idx

    # INTERIOR POINTS ########################################################

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """Returns the bounding box of the cell"""
        if not self.is_parameterized():
            raise NotParameterizedError("getting bounding box")
        x1, x2 = self.get_boundary_points()
        xmin: float = min(x1)
        xmax: float = max(x1)
        ymin: float = min(x2)
        ymax: float = max(x2)
        return xmin, xmax, ymin, ymax

    def is_in_interior(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Returns a boolean array indicating whether each point is in the interior
        """
        if not self.is_parameterized():
            raise NotParameterizedError("checking if points are in interior")
        is_in = np.zeros(np.shape(x), dtype=bool)
        # check if points are in outer boundary
        is_in = self.components[0].is_in_interior_contour(x, y)
        # check if points are in any of the holes
        for i in range(1, self.num_holes + 1):
            is_in = is_in & ~self.components[i].is_in_interior_contour(x, y)
        return is_in

    def get_distance_to_boundary(self, x: float, y: float) -> float:
        """Returns the distance to the boundary at each point"""
        if not self.is_parameterized():
            raise NotParameterizedError("getting distance to boundary")
        dist = float("inf")
        for i in range(self.num_holes + 1):
            dist = min(dist, self.components[i].get_distance_to_boundary(x, y))
        return dist

    def set_interior_mesh_size(self, rows: int, cols: int) -> None:
        """Set the size of the mesh of interior points"""
        self.int_mesh_size = (rows, cols)

    def set_interior_point_tolerance(
        self, rtol: float = 0.02, atol: float = 0.02
    ) -> None:
        """
        Set the minimum distance to the boundary for sampled interior points.

        """
        msg = " must be a positive number"
        if not isinstance(atol, (float, int)):
            raise TypeError("atol" + msg)
        if atol <= 0:
            raise TypeError("atol" + msg)
        self.atol = atol
        if not isinstance(rtol, (float, int)):
            raise TypeError("rtol" + msg)
        if rtol <= 0:
            raise TypeError("rtol" + msg)
        self.rtol = rtol

    def generate_interior_points(self) -> None:
        """
        Returns (x, y, is_inside) where x,y are a meshgrid covering the
        cell K, and is_inside is a boolean array that is True for
        interior points
        """

        rows, cols = self.int_mesh_size

        # find region of interest
        xmin, xmax, ymin, ymax = self.get_bounding_box()

        # set up grid
        x_coord = np.linspace(xmin, xmax, rows)
        y_coord = np.linspace(ymin, ymax, cols)
        self.int_x1, self.int_x2 = np.meshgrid(x_coord, y_coord)

        # determine which points are inside K
        self.is_inside = self.is_in_interior(self.int_x1, self.int_x2)

        # set minimum desired distance to the boundary
        h = min([xmax - xmin, ymax - ymin])
        min_dist_to_bdy = max([self.atol, self.rtol * h])

        # ignore points too close to the boundary
        for i in range(rows):
            for j in range(cols):
                if self.is_inside[i, j]:
                    d = self.get_distance_to_boundary(
                        self.int_x1[i, j], self.int_x2[i, j]
                    )
                    if d < min_dist_to_bdy:
                        self.is_inside[i, j] = False

    # FUNCTION EVALUATION ####################################################
    def evaluate_function_on_boundary(self, fun: Func_R2_R) -> np.ndarray:
        """Return fun(x) for each sampled point on contour"""
        if not self.is_parameterized():
            raise NotParameterizedError("evaluating function on boundary")
        vals = np.zeros((self.num_pts,))
        for i in range(self.num_holes + 1):
            j = self.component_start_idx[i]
            jp1 = self.component_start_idx[i + 1]
            vals[j:jp1] = self.components[i].evaluate_function_on_contour(fun)
        return vals

    def get_boundary_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the x1 and x2 coordinates of the boundary points"""
        if not self.is_parameterized():
            raise NotParameterizedError("getting boundary points")
        x1 = np.zeros((self.num_pts,))
        x2 = np.zeros((self.num_pts,))
        for i in range(self.num_holes + 1):
            j = self.component_start_idx[i]
            jp1 = self.component_start_idx[i + 1]
            x1[j:jp1], x2[j:jp1] = self.components[i].get_sampled_points()
        return x1, x2

    def get_unit_tangent(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the components of the unit tangent vector"""
        zeros = np.zeros((self.num_pts,))
        ones = np.ones((self.num_pts,))
        t1 = self.dot_with_tangent(ones, zeros)
        t2 = self.dot_with_tangent(zeros, ones)
        return t1, t2

    def get_unit_normal(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the components of the unit normal vector"""
        zeros = np.zeros((self.num_pts,))
        ones = np.ones((self.num_pts,))
        n1 = self.dot_with_normal(ones, zeros)
        n2 = self.dot_with_normal(zeros, ones)
        return n1, n2

    def get_dx_norm(self) -> np.ndarray:
        """Returns the norm of the first derivative"""
        ones = np.ones((self.num_pts,))
        return self.multiply_by_dx_norm(ones)

    def dot_with_tangent(
        self, comp1: np.ndarray, comp2: np.ndarray
    ) -> np.ndarray:
        """Returns the dot product (comp1, comp2) * unit_tangent"""
        if not self.is_parameterized():
            raise NotParameterizedError("dotting with tangent")
        if len(comp1) != self.num_pts or len(comp2) != self.num_pts:
            raise SizeMismatchError(
                "comp1 and comp2 must be same length as boundary"
            )
        res = np.zeros((self.num_pts,))
        for i in range(self.num_holes + 1):
            j = self.component_start_idx[i]
            jp1 = self.component_start_idx[i + 1]
            res[j:jp1] = self.components[i].dot_with_tangent(
                comp1[j:jp1], comp2[j:jp1]
            )
        return res

    def dot_with_normal(
        self, comp1: np.ndarray, comp2: np.ndarray
    ) -> np.ndarray:
        """Returns the dot product (comp1, comp2) * unit_normal"""
        if not self.is_parameterized():
            raise NotParameterizedError("dotting with normal")
        if len(comp1) != self.num_pts or len(comp2) != self.num_pts:
            raise SizeMismatchError(
                "comp1 and comp2 must be same length as boundary"
            )
        res = np.zeros((self.num_pts,))
        for i in range(self.num_holes + 1):
            j = self.component_start_idx[i]
            jp1 = self.component_start_idx[i + 1]
            res[j:jp1] = self.components[i].dot_with_normal(
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
        for i in range(self.num_holes + 1):
            j = self.component_start_idx[i]
            jp1 = self.component_start_idx[i + 1]
            vals_dx_norm[j:jp1] = self.components[i].multiply_by_dx_norm(
                vals[j:jp1]
            )
        return vals_dx_norm

    # INTEGRATION ############################################################
    def integrate_over_boundary(self, vals: np.ndarray) -> float:
        """Integrate vals over the boundary"""
        if not self.is_parameterized():
            raise NotParameterizedError("integrating over boundary")
        vals_dx_norm = self.multiply_by_dx_norm(vals)
        return self.integrate_over_boundary_preweighted(vals_dx_norm)

    def integrate_over_boundary_preweighted(
        self, vals_dx_norm: np.ndarray
    ) -> float:
        """Integrate vals over the boundary without multiplying by dx_norm"""
        if not self.is_parameterized():
            raise NotParameterizedError("integrating over boundary")
        if len(vals_dx_norm) != self.num_pts:
            raise SizeMismatchError("vals must be same length as boundary")

        # NOTE: numpy.sum() is more stable, but this uses more memory
        y = np.zeros((self.num_pts,))
        for i in range(self.num_holes + 1):
            c = self.components[i]
            h = 2 * np.pi * c.num_edges / c.num_pts
            j = self.component_start_idx[i]
            jp1 = self.component_start_idx[i + 1]
            y[j:jp1] = h * vals_dx_norm[j:jp1]
        return float(np.sum(y))
