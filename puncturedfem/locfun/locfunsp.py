"""
locfunsp.py
===========

Module containing the LocalFunctionSpace class, which is used to represent a
basis of the local Poisson space V_p(K) on a mesh cell K.
"""

from typing import Optional

from tqdm import tqdm

from ..mesh.cell import MeshCell
from ..solver.globkey import GlobalKey
from .edge_space import EdgeSpace
from .locfun import LocalFunction, Polynomial
from .nystrom import NystromSolver


class LocalFunctionSpace:
    """
    A collection of local functions LocalFunction objects) that form a basis of
    the local Poisson space V_p(K). TheLocalFunctions are partitioned into three
    types:
            vert_funs: vertex functions (harmonic, trace supported on two edges)
            edge_funs: Edge functions (harmonic, trace supported on one Edge)
            bubb_funs: bubble functions (Polynomial Laplacian, zero trace)
    In the case where the mesh cell K has a vertex-free Edge (that is, the Edge
    is a simple closed contour, e.g. a circle), no vertex functions are
    associated with that Edge.
    """

    deg: int
    num_vert_funs: int
    num_edge_funs: int
    num_bubb_funs: int
    num_funs: int
    vert_funs: list[LocalFunction]
    edge_funs: list[LocalFunction]
    bubb_funs: list[LocalFunction]
    nyst: NystromSolver

    def __init__(
        self,
        K: MeshCell,
        edge_spaces: Optional[list[EdgeSpace]] = None,
        deg: int = 1,
        compute_interior_values: bool = True,
        compute_interior_gradient: bool = False,
        verbose: bool = True,
        processes: int = 1,
    ) -> None:
        """
        Initialize the LocalFunctionSpace object for a given MeshCell K and the
        Edge spaces, which is a list of EdgeSpace objects for each Edge in K, in
        the order that the edges appear in K.edges. The Edge spaces must be
        computed before theLocalFunctionSpace object is initialized. This
        initialization computes all function metadata (e.g. traces, interior
        values) and and interior values.

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

        # bubble functions: zero trace, Polynomial Laplacian
        self.build_bubble_funs()

        # build vertex functions...')
        self.build_vert_funs(edge_spaces)

        # build Edge functions
        self.build_edge_funs(edge_spaces)

        # count number of each type of function
        self.compute_num_funs()

        # compute all function metadata
        self.compute_all(verbose=verbose, processes=processes)

        # find interior values
        if compute_interior_values or compute_interior_gradient:
            self.find_interior_values(
                verbose=verbose, compute_grad=compute_interior_gradient
            )

    def set_deg(self, deg: int) -> None:
        """
        Set degree of Polynomial space
        """
        if not isinstance(deg, int):
            raise TypeError("deg must be an integer")
        if deg < 1:
            raise ValueError("deg must be a positive integer")
        self.deg = deg

    def find_interior_values(
        self, verbose: bool = True, compute_grad: bool = False
    ) -> None:
        """
        Equivalent to running v.compute_interior_values() for each
        LocalFunction v
        """
        if verbose:
            print("Finding interior values...")
            for v in tqdm(self.get_basis()):
                v.compute_interior_values(compute_grad)
        else:
            for v in self.get_basis():
                v.compute_interior_values(compute_grad)

    # BUILD FUNCTIONS ########################################################

    def compute_num_funs(self) -> None:
        """
        Sum the number of vertex, Edge, and bubble functions
        """
        self.num_vert_funs = len(self.vert_funs)
        self.num_edge_funs = len(self.edge_funs)
        self.num_bubb_funs = len(self.bubb_funs)
        self.num_funs = (
            self.num_vert_funs + self.num_edge_funs + self.num_bubb_funs
        )

    def compute_all(self, verbose: bool = True, processes: int = 1) -> None:
        """
        Equivalent to running v.compute_all(K) for eachLocalFunction v.
        """
        if processes == 1:
            self.compute_all_sequential(verbose=verbose)
        elif processes > 1:
            self.compute_all_parallel(verbose=verbose, processes=processes)
        else:
            raise ValueError("processes must be a positive integer")

    def compute_all_sequential(self, verbose: bool = True) -> None:
        """
        Equivalent to running v.compute_all(K) for eachLocalFunction v.
        """
        if verbose:
            print("Computing function metadata...")
            for v in tqdm(self.get_basis()):
                v.compute_all()
        else:
            for v in self.get_basis():
                v.compute_all()

    def compute_all_parallel(
        self, verbose: bool = True, processes: int = 1
    ) -> None:
        """
        Equivalent to running v.compute_all(K) for eachLocalFunction v, using
        multiprocessing to parallelize computation.
        """
        raise NotImplementedError("Parallel computation not yet implemented")

    def build_bubble_funs(self) -> None:
        """Construct bubble functions"""
        # bubble functions
        num_bubb = (self.deg * (self.deg - 1)) // 2
        self.bubb_funs = []
        for k in range(num_bubb):
            v_key = GlobalKey(fun_type="bubb", bubb_space_idx=k)
            v = LocalFunction(nyst=self.nyst, key=v_key)
            p = Polynomial()
            p.add_monomial_with_idx(coef=-1.0, idx=k)
            v.set_laplacian_polynomial(p)
            self.bubb_funs.append(v)

    def build_vert_funs(self, edge_spaces: list[EdgeSpace]) -> None:
        """Construct vertex functions from Edge spaces"""

        # find all Vertices on MeshCell
        vert_idx_set = set()
        for c in self.nyst.K.components:
            for e in c.edges:
                if not e.is_loop:
                    vert_idx_set.add(e.anchor.idx)
                    vert_idx_set.add(e.endpnt.idx)
        vert_keys: list[GlobalKey] = []
        for vert_idx in vert_idx_set:
            vert_keys.append(GlobalKey(fun_type="Vert", vert_idx=vert_idx))

        # initialize list of vertex functions and set traces
        self.vert_funs = []
        for vert_key in vert_keys:
            v = LocalFunction(nyst=self.nyst, key=vert_key)
            for j, b in enumerate(edge_spaces):
                for k in range(b.num_vert_funs):
                    if b.vert_fun_global_keys[k].vert_idx == vert_key.vert_idx:
                        v.poly_trace.polys[j] = b.vert_fun_traces[k]
            self.vert_funs.append(v)

    def build_edge_funs(self, edge_spaces: list[EdgeSpace]) -> None:
        """Construct Edge functions from Edge spaces"""

        # initialize list of Edge functions
        self.edge_funs = []

        # loop over edges on MeshCell
        for b in edge_spaces:
            # locate Edge within MeshCell
            glob_edge_idx = b.e.idx
            glob_edge_idx_list = [e.idx for e in self.nyst.K.get_edges()]
            edge_idx = glob_edge_idx_list.index(glob_edge_idx)

            # loop over Edge functions
            for k in range(b.num_edge_funs):
                v_trace = b.edge_fun_traces[k]

                # create harmonicLocalFunction
                v = LocalFunction(nyst=self.nyst, key=b.edge_fun_global_keys[k])

                # set Dirichlet data
                v.poly_trace.polys[edge_idx] = v_trace

                # add to list of Edge functions
                self.edge_funs.append(v)

    def get_basis(self) -> list[LocalFunction]:
        """Return list of all functions"""
        return self.vert_funs + self.edge_funs + self.bubb_funs
