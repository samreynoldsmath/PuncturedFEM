import copy
import numpy as np

from .. import quad

class edge:

    __slots__ = (
        'id',
        'etype',
        'qtype',
        'num_pts',
        'x',
        'unit_tangent',
        'unit_normal',
        'dx_norm',
        'curvature',
    )

    def __init__(self, etype: str, q: quad.quad, id: any = [], **kwargs):

        # optional identifier (for use in global mesh)
        self.id = id

        # label edge and quadrature types
        self.etype = etype
        self.qtype = q.type

        # record the number of sampled points
        self.num_pts = 2 * q.n + 1

        # import edgelib object of this edge type:
        # assumes a file called <self.etype>.py exists in mesh/edgelib
        # this file should contain definitions for _x(), _dx(), and _ddx()
        e = __import__(
            f'puncturedfem.mesh.edgelib.{self.etype}',
            fromlist=f'mesh.edgelib.{self.etype}'
        )

        # compute and store points on the boundary
        self.x = e._x(q.t, **kwargs)

        # unweighted square norm of derivative
        dx = e._dx(q.t, **kwargs)
        dx2 = dx[0,:] ** 2 + dx[1,:] ** 2

        # norm of derivative (with chainrule)
        self.dx_norm = np.sqrt(dx2) * q.wgt

        # unit tangent vector
        self.unit_tangent = dx / np.sqrt(dx2)

        # outward unit normal vector
        self._set_unit_normal()

        # get signed curvature
        ddx = e._ddx(q.t, **kwargs)
        self.curvature = (
            ddx[0, :] * self.unit_normal[0, :] +
            ddx[1, :] * self.unit_normal[1, :] ) / dx2

    def duplicate(self):
        return copy.deepcopy(self)

    def evaluate_function(self, fun: callable, ignore_endpoint=False):
        """
        Return fun(x) for each sampled point on edge
        """
        if ignore_endpoint:
            k = 1
        else:
            k = 0
        y = np.zeros((self.num_pts - k,))
        for i in range(self.num_pts - k):
            y[i] = fun(self.x[:, i])
        return y

    def reverse_orientation(self) -> None:
        """
        Reverse the orientation of this edge using the reparameterization
        x(2 pi - t). The chain rule flips the sign of some derivative-based
        quanitites.
        """
        # vector quantities
        self.x = np.fliplr(self.x)
        self.unit_tangent = - np.fliplr(self.unit_tangent)
        self.unit_normal = - np.fliplr(self.unit_normal)

        # scalar quantities
        self.dx_norm = np.flip(self.dx_norm)
        self.curvature = - np.flip(self.curvature)

    def join_points(self, a, b) -> None:
        """
        Join the points a to b with this edge.
        Throws an error if this edge is a closed contour.
        """
        TOL = 1e-12

        # check that specified endpoints are distinct
        ab_norm = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        if ab_norm < TOL:
            raise Exception('a and b must be distinct points')

        # check that endpoints of edge are distinct
        x = self.x[:, 0]
        y = self.x[:, self.num_pts-1]
        xy_norm = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
        if xy_norm < TOL:
            raise Exception('edge must have distinct endpoints')

        # anchor starting point to origin
        self.translate(-x)

        # rotate
        theta = - np.arctan2(y[1] - x[1], y[0] - x[0])
        theta += np.arctan2(b[1] - a[1], b[0] - a[0])
        theta *= 180 / np.pi
        self.rotate(theta)

        # rescale
        alpha = ab_norm / xy_norm
        self.dialate(alpha)

        # anchor at point a
        self.translate(a)

    def translate(self, a) -> None:
        """
        Translate by (a[0], a[1])
        """
        self.x[0,:] += a[0]
        self.x[1,:] += a[1]

    def dialate(self, alpha: float) -> None:
        """
        Dialate by a scalar alpha
        """

        if np.abs(alpha) < 1e-12:
            raise Exception('Dialation factor alpha must be nonzero')

        self.x *= alpha
        self.dx_norm *= np.abs(alpha)
        self.curvature *= 1 / alpha

    def rotate(self, theta: float) -> None:
        """
        Rotate counterclockwise by theta (degrees)
        """

        if theta % 360 == 0:
            return None

        c = np.cos(theta * np.pi / 180)
        s = np.sin(theta * np.pi / 180)
        R = np.array([ [c, -s], [s, c] ])

        self.apply_orthogonal_transformation(R)

    def reflect_across_x_axis(self) -> None:
        """
        Reflect across the horizontal axis
        """
        A = np.array([ [1, 0], [0, -1] ])
        self.apply_orthogonal_transformation(A)

    def reflect_across_y_axis(self) -> None:
        """
        Reflect across the vertical axis
        """
        A = np.array([ [-1, 0], [0, 1] ])
        self.apply_orthogonal_transformation(A)

    def apply_orthogonal_transformation(self, A) -> None:
        """
        Transforms 2-dimensional space with the linear map
            x \mapsto A * x
        where A is a 2 by 2 orthogonal matrix, i.e. A^T * A = I

        It is important that A is orthogonal, since the first derivative norm
        as well as the curvature are invariant under such a transformation.
        """

        # safety checks
        TOL = 1e-12
        msg = 'A must be a 2 by 2 orthogonal matrix'
        if np.shape(A) != (2,2):
            raise Exception(msg)
        if np.linalg.norm(np.transpose(A) @ A - np.eye(2)) > TOL:
            raise Exception(msg)

        # apply transformation to vector quantities
        self.x = A @ self.x
        self.unit_tangent = A @ self.unit_tangent
        self._set_unit_normal()

        # determine if the sign of curvature has flipped
        a = A[0,0]
        b = A[0,1]
        c = A[1,0]
        d = A[1,1]
        if np.abs(b - c) < TOL and np.abs(a + d) < TOL:
            self.curvature *= -1

    def _set_unit_normal(self) -> None:
        self.unit_normal = np.zeros((2, self.num_pts))
        self.unit_normal[0, :] = self.unit_tangent[1, :]
        self.unit_normal[1, :] = - self.unit_tangent[0, :]

    def __eq__(self, other) -> bool:
        TOL = 1e-12
        return np.linalg.norm(self.x - other.x) < TOL

    def __repr__(self) -> str:
        msg = ''
        msg += f'id:        {self.id}\n'
        msg += f'etype:     {self.etype}\n'
        msg += f'qtype:     {self.qtype}\n'
        msg += f'num_pts:   {self.num_pts}\n'
        return msg
