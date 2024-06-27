"""
Representation of a basis of the local Poisson space V_p(K).

Classes
-------
LocalPoissonSpace
    A basis of the local Poisson space V_p(K).
"""

from typing import Optional, Union

from deprecated import deprecated
from tqdm import tqdm
import numpy as np

from ..mesh.cell import MeshCell
from ..solver.globkey import GlobalKey
from .edge_space import EdgeSpace
from .local_poisson import LocalPoissonFunction
from .nystrom import NystromSolver
from .poly.poly import Polynomial
from .trace import DirichletTrace


class LocalPoissonSpace:
    """
    A basis of the local Poisson space V_p(K).

    A collection of LocalPoissonFunction objects that form a basis of the local
    Poisson space V_p(K). The functions are partitioned into three types:
        vert_funs: vertex functions (harmonic, trace supported on two edges)
        edge_funs: Edge functions (harmonic, trace supported on one Edge)
        bubb_funs: bubble functions (Polynomial Laplacian, zero trace)
    In the case where the mesh cell K has a vertex-free edge (that is, the edge
    is a simple closed contour, e.g. a circle), no vertex functions are
    associated with that edge. In this case, the edge functions are the only
    functions associated with the edge, which are simply traces of polynomials.

    Attributes
    ----------
    deg : int
        Degree of local Poisson space.
    num_vert_funs : int
        Number of vertex functions.
    num_edge_funs : int
        Number of edge functions.
    num_bubb_funs : int
        Number of bubble functions.
    num_funs : int
        Total number of functions.
    vert_funs : list[LocalPoissonFunction]
        List of vertex functions.
    edge_funs : list[LocalPoissonFunction]
        List of edge functions.
    bubb_funs : list[LocalPoissonFunction]
        List of bubble functions.
    nyst : NystromSolver
        Nystrom solver object for solving integral equations.
    """

    deg: int
    num_vert_funs: int
    num_edge_funs: int
    num_bubb_funs: int
    num_funs: int
    vert_funs: list[LocalPoissonFunction]
    edge_funs: list[LocalPoissonFunction]
    bubb_funs: list[LocalPoissonFunction]
    nyst: NystromSolver
    centroid: tuple[float, float]
    area: float
    compute_interior_values: bool
    compute_interior_gradient: bool

    def __init__(
        self,
        K: MeshCell,
        edge_spaces: Optional[list[EdgeSpace]] = None,
        deg: int = 1,
        compute_interior_values: bool = True,
        compute_interior_gradient: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the LocalFunctionSpace object.

        Parameters
        ----------
        K : MeshCell
            Mesh MeshCell
        edge_spaces : list[EdgeSpace], optional
            List of EdgeSpace objects for each Edge in K
        deg : int, optional
            Degree of polynomial space, by default 1
        verbose : bool, optional
            Print progress, by default True
        processes : int, optional
            Number of processes to use for parallel computation, by default 1
        """
        # set degree of Polynomial space
        self.set_deg(deg)

        self.compute_interior_values = compute_interior_values
        self.compute_interior_gradient = compute_interior_gradient

        self._build_centroid(K)
        self._find_area(K)

        # construct edge spaces, if not provided
        if edge_spaces is None:
            edge_spaces = []
            for e in K.get_edges():
                edge_spaces.append(EdgeSpace(e, self.deg))

        # set up Nyström solver
        self.nyst = NystromSolver(K, verbose=verbose)

        # build each type of function
        self._build_basis(edge_spaces, verbose=verbose)

    def set_deg(self, deg: int) -> None:
        """
        Set degree of Polynomial space.

        Parameters
        ----------
        deg : int
            Degree of Polynomial space.
        """
        if not isinstance(deg, int):
            raise TypeError("deg must be an integer")
        if deg < 1:
            raise ValueError("deg must be a positive integer")
        self.deg = deg

    def get_basis(self) -> list[LocalPoissonFunction]:
        """
        Get the list of all basis functions.

        Returns
        -------
        list[LocalPoissonFunction]
            List of all basis functions.
        """
        return self.vert_funs + self.edge_funs + self.bubb_funs

    def _build_centroid(self, K: MeshCell) -> None:
        x1, x2 = K.get_boundary_points()
        centroid_x = np.mean(x1)
        centroid_y = np.mean(x2)
        self.centroid = (centroid_x, centroid_y)

    def _find_area(self, K: MeshCell) -> None:
        x1, x2 = K.get_boundary_points()
        xn = K.dot_with_normal(x1, x2)
        self.area = 0.5 * K.integrate_over_boundary(xn)

    def _compute_num_funs(self) -> None:
        self.num_vert_funs = len(self.vert_funs)
        self.num_edge_funs = len(self.edge_funs)
        self.num_bubb_funs = len(self.bubb_funs)
        self.num_funs = (
            self.num_vert_funs + self.num_edge_funs + self.num_bubb_funs
        )

    def _build_basis(
        self,
        edge_spaces: list[EdgeSpace],
        verbose: bool,
    ) -> None:
        self._build_bubb_funs(verbose)
        self._build_vert_funs(edge_spaces, verbose)
        self._build_edge_funs(edge_spaces, verbose)
        self._compute_num_funs()

    def _build_bubb_funs(self, verbose: bool) -> None:
        # Bubble functions are zero on the boundary and have a polynomial
        # Laplacian.
        self.bubb_funs = []
        if self.deg < 2:
            return
        num_bubb = (self.deg * (self.deg - 1)) // 2
        range_num_bubb: Union[tqdm, range]
        if verbose:
            range_num_bubb = tqdm(
                range(num_bubb), desc="Building bubble functions"
            )
        else:
            range_num_bubb = range(num_bubb)
        for k in range_num_bubb:
            v_key = GlobalKey(fun_type="bubb", bubb_space_idx=k)
            p = self._build_centered_monomial(k)
            self.bubb_funs.append(
                LocalPoissonFunction(
                    nyst=self.nyst,
                    laplacian=p,
                    key=v_key,
                    evaluate_interior=self.compute_interior_values,
                    evaluate_gradient=self.compute_interior_gradient,
                )
            )

    def _build_centered_monomial(self, idx: int) -> Polynomial:
        m1 = Polynomial([(1.0, 1, 0)]) - self.centroid[0]
        m2 = Polynomial([(1.0, 0, 1)]) - self.centroid[1]
        p = Polynomial()
        p.add_monomial_with_idx(coef=-1 / self.area**2, idx=idx)
        p = p.compose(m1, m2)
        return p

    def _build_vert_funs(
        self, edge_spaces: list[EdgeSpace], verbose: bool
    ) -> None:
        # Vertex functions are harmonic and have trace supported on two edges,
        # with the common vertex having a value of 1 and all other vertices
        # having a value of 0.
        vert_idx_set = set()
        for c in self.nyst.K.components:
            for e in c.edges:
                if not e.is_loop:
                    vert_idx_set.add(e.anchor.idx)
                    vert_idx_set.add(e.endpnt.idx)
        vert_keys: list[GlobalKey] = []
        for vert_idx in vert_idx_set:
            vert_keys.append(GlobalKey(fun_type="vert", vert_idx=vert_idx))

        self.vert_funs = []

        if not vert_keys:
            return

        # initialize list of vertex functions and set traces
        vert_keys_iter: Union[tqdm, list[GlobalKey]]
        if verbose:
            vert_keys_iter = tqdm(vert_keys, desc="Building vertex functions")
        else:
            vert_keys_iter = vert_keys
        for vert_key in vert_keys_iter:
            v_trace = DirichletTrace(
                edges=self.nyst.K.get_edges(), funcs=lambda x, y: 0
            )
            for j, b in enumerate(edge_spaces):
                for k in range(b.num_vert_funs):
                    if b.vert_fun_global_keys[k].vert_idx == vert_key.vert_idx:
                        v_trace.set_func_from_poly_on_edge(
                            edge_index=j, poly=b.vert_fun_traces[k]
                        )
            v = LocalPoissonFunction(
                nyst=self.nyst,
                trace=v_trace,
                key=vert_key,
                evaluate_interior=self.compute_interior_values,
                evaluate_gradient=self.compute_interior_gradient,
            )
            self.vert_funs.append(v)

    def _build_edge_funs(
        self, edge_spaces: list[EdgeSpace], verbose: bool
    ) -> None:
        # Edge functions are harmonic and have trace supported on one edge.
        edge_spaces_iter: Union[tqdm, list[EdgeSpace]]
        if verbose:
            edge_spaces_iter = tqdm(
                edge_spaces, desc="Building edge functions  "
            )
        else:
            edge_spaces_iter = edge_spaces

        self.edge_funs = []
        for b in edge_spaces_iter:
            # locate Edge within MeshCell
            glob_edge_idx = b.e.idx
            glob_edge_idx_list = [e.idx for e in self.nyst.K.get_edges()]
            edge_idx = glob_edge_idx_list.index(glob_edge_idx)

            # loop over edge functions
            for k in range(b.num_edge_funs):
                v_trace = DirichletTrace(
                    edges=self.nyst.K.get_edges(),
                    funcs=lambda x, y: 0,
                )
                v_trace.set_func_from_poly_on_edge(
                    edge_index=edge_idx, poly=b.edge_fun_traces[k]
                )
                v = LocalPoissonFunction(
                    nyst=self.nyst,
                    trace=v_trace,
                    key=b.edge_fun_global_keys[k],
                    evaluate_interior=self.compute_interior_values,
                    evaluate_gradient=self.compute_interior_gradient,
                )
                self.edge_funs.append(v)


@deprecated(version="0.5.0", reason="Use LocalPoissonSpace instead")
class LocalFunctionSpace(LocalPoissonSpace):
    """
    Use LocalPoissonSpace. Deprecated in version 0.5.0.
    """

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
