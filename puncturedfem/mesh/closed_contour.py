from matplotlib import path
from numpy import argmax, array, inf, linspace, meshgrid, shape, sqrt, zeros

from .bounding_box import get_bounding_box
from .edge import NotParameterizedError, edge
from .vert import vert


class closed_contour:
    """
    List of edges forming a closed contour, assumed to be simple
    with edges listed successively.
    """

    edges: list[edge]
    num_edges: int
    edge_orient: list[int]
    interior_point: vert
    num_pts: int
    local_vert_idx: list[int]
    closest_vert_idx: list[int]

    def __init__(
        self,
        cell_id: int,
        edges: list[edge] = None,
        edge_orients: list[int] = None,
    ) -> None:
        self.set_cell_id(cell_id)
        self.num_edges = 0
        self.edges = []
        self.edge_orients = []
        self.add_edges(edges, edge_orients)

    def set_cell_id(self, cell_id: int) -> None:
        if not isinstance(cell_id, int):
            raise TypeError(
                f"cell_id = {cell_id} invalid, must be a positive integer"
            )
        if cell_id < 0:
            raise ValueError(
                f"cell_id = {cell_id} invalid, must be a positive integer"
            )
        self.cell_id = cell_id

    # EDGE MANAGMENT #########################################################
    def add_edge(self, e: edge, edge_orient: int) -> None:
        """Add edge to contour"""
        if edge_orient != +1 and edge_orient != -1:
            raise ValueError("Orientation must be +1 or -1")
        if e in self.edges:
            return
        self.edges.append(e)
        self.edge_orients.append(edge_orient)
        self.num_edges += 1

    def add_edges(self, edges: list[edge], edge_orients: list[int]) -> None:
        """Add edges to contour"""
        if edges is None:
            return
        if len(edges) != len(edge_orients):
            raise ValueError("Must provide orientation for each edge")
        for e, o in zip(edges, edge_orients):
            self.add_edge(e, o)

    def is_closed(self) -> bool:
        """Returns true if the contour is closed"""
        raise NotImplementedError()

    # PARAMETERIZATON ########################################################
    def is_parameterized(self) -> bool:
        return all([e.is_parameterized for e in self.edges])

    def parameterize(self, quad_dict: dict):
        """Parameterize each edge"""
        # TODO: eliminate redundant calls to parameterize
        for i in range(self.num_edges):
            self.edges[i].parameterize(quad_dict)
            if self.edge_orients[i] == -1:
                self.edges[i].reverse_orientation()
        self.find_num_pts()
        self.find_local_vert_idx()
        self.find_closest_local_vertex_index()
        self.find_interior_point()

    def deparameterize(self):
        for e in self.edges:
            e.deparameterize()
        self.num_pts = 0
        self.closest_vert_idx = None

    def find_num_pts(self) -> None:
        """Record the total number of sampled points on the boundary"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding num_pts")
        self.num_pts = 0
        for e in self.edges:
            self.num_pts += e.num_pts - 1

    def find_local_vert_idx(self) -> None:
        """Get the index of the starting point of each edge"""
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
        mid_idx = zeros((self.num_edges,), dtype=int)
        for i in range(self.num_edges):
            n = self.edges[i].num_pts // 2  # 2n points per edge
            mid_idx[i] = self.vert_idx[i] + n

        # on first half of an edge, the closest vertex is the starting
        # point on that edge; on the second half of an edge, the closest vertex
        # is the starting point of the next edge
        self.closest_vert_idx = zeros((self.num_pts,), dtype=int)
        for i in range(self.num_edges):
            self.closest_vert_idx[
                self.vert_idx[i] : mid_idx[i]
            ] = self.vert_idx[i]
            self.closest_vert_idx[mid_idx[i] : self.vert_idx[i + 1]] = (
                self.vert_idx[i + 1] % self.num_pts
            )

    # INTERIOR POINTS ########################################################
    def get_distance_to_boundary(self, x, y) -> float:
        """Minimum distance from (x,y) to a point on the boundary"""
        if not self.is_parameterized():
            raise NotParameterizedError("finding distance to boundary")
        dist = inf
        for e in self.edges:
            dist2e = min((e.x[0, :] - x) ** 2 + (e.x[1, :] - y) ** 2)
            dist = min([dist, dist2e])
        return sqrt(dist)

    def is_in_interior_contour(self, x: array, y: array):
        if x.shape != y.shape:
            raise Exception("x and y must have same size")

        is_inside = zeros(x.shape, dtype=bool)
        x1, x2 = self.get_sampled_points()
        p = path.Path(array([x1, x2]).transpose())

        if len(x.shape) == 1:
            M = x.shape[0]
            for i in range(M):
                is_inside[i] = p.contains_point([x[i], y[i]])
        elif len(x.shape) == 2:
            M, N = x.shape
            for i in range(M):
                for j in range(N):
                    is_inside[i, j] = p.contains_point([x[i, j], y[i, j]])

        return is_inside

    def get_bounding_box_cell(self):
        x1, x2 = self.get_boundary_points()
        return get_bounding_box(x=x1, y=x2)

    def find_interior_point(self) -> vert:
        """Returns an interior point."""
        # TODO: Uses a brute force search. There is likely a more efficient way.

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
            x_coord = linspace(xmin, xmax, M)
            y_coord = linspace(ymin, ymax, N)
            x, y = meshgrid(x_coord, y_coord)

            # determine which points are in the interior
            is_inside = self.is_in_interior_contour(x, y)

            # for each interior point in grid, compute distance to the boundary
            dist = zeros(shape(x))
            for i in range(M):
                for j in range(N):
                    if is_inside[i, j]:
                        dist[i, j] = self.get_distance_to_boundary(
                            x[i, j], y[i, j]
                        )

            # pick a point farthest from the boundary
            k = argmax(dist, keepdims=True)
            ii = k[0][0] // M
            jj = k[0][0] % M
            d = dist[ii, jj]

            # if the best candidate is too close to the boundary,
            # refine grid and search again
            M = 4 * (M // 2) + 1
            N = 4 * (N // 2) + 1

            if M * N > 1_000_000:
                raise Exception("Unable to locate an interior point")

            self.interior_point = vert(x=x[ii, jj], y=y[ii, jj])

    # FUNCTION EVALUATION ####################################################
    def evaluate_function_on_contour(self, fun: callable):
        """Return fun(x) for each sampled point on contour"""
        if not self.is_parameterized():
            raise NotParameterizedError("evaluating function on contour")
        y = zeros((self.num_pts,))
        for j in range(self.num_edges):
            y[self.vert_idx[j] : self.vert_idx[j + 1]] = self.edges[
                j
            ].evaluate_function(fun, ignore_endpoint=True)
        return y

    def get_sampled_points(self) -> tuple[array, array]:
        """Returns the x1 and x2 coordinates of the boundary points"""
        if not self.is_parameterized():
            raise NotParameterizedError("getting boundary points")
        x1 = self.evaluate_function_on_contour(lambda x: x[0])
        x2 = self.evaluate_function_on_contour(lambda x: x[1])
        return x1, x2

    def dot_with_tangent(self, v1, v2):
        """Returns the dot product (v1, v2) * unit_tangent"""
        if not self.is_parameterized:
            raise NotParameterizedError("dotting with tangent")
        res = zeros((self.num_pts,))
        for i in range(self.num_edges):
            j = self.vert_idx[i]
            jp1 = self.vert_idx[i + 1]
            res[j:jp1] = self.edges[i].dot_with_tangent(v1[j:jp1], v2[j:jp1])
        return res

    def dot_with_normal(self, v1, v2):
        """Returns the dot product (v1, v2) * unit_normal"""
        if not self.is_parameterized:
            raise NotParameterizedError("dotting with normal")
        res = zeros((self.num_pts,))
        for i in range(self.num_edges):
            j = self.vert_idx[i]
            jp1 = self.vert_idx[i + 1]
            res[j:jp1] = self.edges[i].dot_with_normal(v1[j:jp1], v2[j:jp1])
        return res

    def multiply_by_dx_norm(self, vals):
        """
        Returns f multiplied against the norm of the derivative of
        the curve parameterization
        """
        if not self.is_parameterized():
            raise NotParameterizedError("multiplying by dx_norm")
        if len(vals) != self.num_pts:
            raise Exception("vals must be same length as boundary")
        vals_dx_norm = zeros((self.num_pts,))
        for i in range(self.num_edges):
            j = self.vert_idx[i]
            jp1 = self.vert_idx[i + 1]
            vals_dx_norm[j:jp1] = self.edges[i].multiply_by_dx_norm(vals[j:jp1])
        return vals_dx_norm

    # INTEGRATION ############################################################
    def integrate_over_closed_contour(self, vals) -> float:
        """Contour integral of vals"""
        if not self.is_parameterized():
            raise NotParameterizedError("integrating over boundary")
        vals_dx_norm = self.multiply_by_dx_norm(vals)
        return self.integrate_over_closed_contour_preweighted(vals_dx_norm)

    def integrate_over_closed_contour_preweighted(self, vals_dx_norm) -> float:
        """Contour integral of vals_dx_norm"""

        # # check inputs
        # if not self.is_parameterized():
        #     raise NotParameterizedError('integrating over boundary')
        # if len(shape(vals_dx_norm)) != 1:
        #     raise Exception('vals_dx_norm must be a vector')
        # if len(vals_dx_norm) != self.num_pts:
        #     raise Exception('vals must be same length as boundary')

        # # if contour is a single edge without a corner, use trapezoid rule
        # if self.num_edges == 1 and self.edges[0].quad_type == 'trap':
        #     vals_vals = zeros((self.edges[0].num_pts,))
        #     vals_vals[:-1] = vals_dx_norm
        #     vals_vals[-1] = vals_dx_norm[0]
        #     res = self.edges[0].integrate_over_edge_preweighted(
        #         vals_vals, ignore_endpoint=False
        #     )

        # # otherwise, use Kress quadrature on each edge
        # else:
        #     res = 0
        #     for i in range(self.num_edges):
        #         j = self.vert_idx[i]
        #         jp1 = self.vert_idx[i + 1]
        #         res += self.edges[i].integrate_over_edge_preweighted(
        #             vals_dx_norm[j:jp1], ignore_endpoint=True)
        # return res

        # TODO: fix this hack
        # numpy.sum() is more stable, but this uses more memory

        from numpy import pi

        y = zeros((self.num_pts,))
        for i in range(self.num_edges):
            h = 2 * pi / (self.edges[i].num_pts - 1)
            y[self.vert_idx[i] : self.vert_idx[i + 1]] = (
                h * vals_dx_norm[self.vert_idx[i] : self.vert_idx[i + 1]]
            )
        return sum(y)
