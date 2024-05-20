"""
Representation of a basis of the local Poisson space V_p(K).

Classes
-------
LocalPoissonSpace
    A basis of the local Poisson space V_p(K).
"""

from typing import Optional

from deprecated import deprecated

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

        # construct edge spaces, if not provided
        if edge_spaces is None:
            edge_spaces = []
            for e in K.get_edges():
                edge_spaces.append(EdgeSpace(e, self.deg))

        # set up Nyström solver
        self.nyst = NystromSolver(K, verbose=verbose)

        # build each type of function
        self._build_bubb_funs()
        self._build_vert_funs(edge_spaces)
        self._build_edge_funs(edge_spaces)
        self._compute_num_funs()

        # find interior values
        if compute_interior_values or compute_interior_gradient:
            raise NotImplementedError(
                "Interior values and gradients are not yet implemented"
            )

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

    def _compute_num_funs(self) -> None:
        self.num_vert_funs = len(self.vert_funs)
        self.num_edge_funs = len(self.edge_funs)
        self.num_bubb_funs = len(self.bubb_funs)
        self.num_funs = (
            self.num_vert_funs + self.num_edge_funs + self.num_bubb_funs
        )

    def _build_bubb_funs(self) -> None:
        # Bubble functions are zero on the boundary and have a polynomial
        # Laplacian.
        if self.deg < 2:
            return
        num_bubb = (self.deg * (self.deg - 1)) // 2
        self.bubb_funs = []
        for k in range(num_bubb):
            v_key = GlobalKey(fun_type="bubb", bubb_space_idx=k)
            p = Polynomial()
            p.add_monomial_with_idx(coef=-1.0, idx=k)
            self.bubb_funs.append(
                LocalPoissonFunction(nyst=self.nyst, laplacian=p, key=v_key)
            )

    def _build_vert_funs(self, edge_spaces: list[EdgeSpace]) -> None:
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

        # initialize list of vertex functions and set traces
        self.vert_funs = []
        for vert_key in vert_keys:
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
                nyst=self.nyst, trace=v_trace, key=vert_key
            )
            self.vert_funs.append(v)

    def _build_edge_funs(self, edge_spaces: list[EdgeSpace]) -> None:
        # Edge functions are harmonic and have trace supported on one edge.
        self.edge_funs = []
        for b in edge_spaces:
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
                    nyst=self.nyst, trace=v_trace, key=b.edge_fun_global_keys[k]
                )
                self.edge_funs.append(v)


@deprecated(version="0.5.0", reason="Use LocalPoissonSpace instead")
class LocalFunctionSpace(LocalPoissonSpace):
    """
    Use LocalPoissonSpace. Deprecated in version 0.5.0.
    """

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
