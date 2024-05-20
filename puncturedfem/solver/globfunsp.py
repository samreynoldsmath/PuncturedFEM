"""
Global function spaces.

Classes
-------
GlobalFunctionSpace
    Represents a global function space V_p(T) on a planar mesh T of degree p.
"""

from tqdm import tqdm

from ..locfun.edge_space import EdgeSpace
from ..locfun.local_space import LocalPoissonSpace
from ..mesh.planar_mesh import PlanarMesh
from ..mesh.quad import QuadDict
from .globkey import GlobalKey


class GlobalFunctionSpace:
    """
    Global function space V_p(T) on a planar mesh T of degree p.

    Attributes
    ----------
    T : PlanarMesh
        Planar mesh
    deg : int
        Degree of polynomial space
    quad_dict : QuadDict
        Dictionary of quadrature rules
    num_funs : int
        Number of functions in the space
    num_vert_funs : int
        Number of vertex functions
    num_edge_funs : int
        Number of edge functions
    num_bubb_funs : int
        Number of bubble functions
    edge_spaces : list[EdgeSpace]
        List of edge spaces
    edge_fun_cumsum : list[int]
        Cumulative sum of the number of edge functions
    cell_dofs : list[list[GlobalKey]]
        List of lists of global keys for each cell
    """

    T: PlanarMesh
    deg: int
    quad_dict: QuadDict
    num_funs: int
    num_vert_funs: int
    num_edge_funs: int
    num_bubb_funs: int
    edge_spaces: list[EdgeSpace]
    edge_fun_cumsum: list[int]
    cell_dofs: list[list[GlobalKey]]

    def __init__(
        self,
        T: PlanarMesh,
        deg: int,
        quad_dict: QuadDict,
        verbose: bool = True,
    ) -> None:
        """
        Initialize a GlobalFunctionSpace object.

        Parameters
        ----------
        T : PlanarMesh
            Planar mesh
        deg : int
            Degree of polynomial space
        quad_dict : dict[str, Quad]
            Dictionary of quadrature rules
        verbose : bool, optional
            If True, print progress bars, by default True
        """
        self.bubb_fun_counter = 0
        self.edge_fun_counter = 0
        self.quad_dict = quad_dict
        self.cell_dofs = [[] for _ in range(T.num_cells)]
        self.set_mesh(T)
        self.set_deg(deg)
        self.build_edge_spaces(verbose=verbose)
        self.compute_dimension()
        self.compute_edge_fun_cumsum()
        if verbose:
            print(self)

    def __str__(self) -> str:
        """Return a string representation of the global function space."""
        s = "GlobalFunctionSpace:"
        s += f"\n\tT: {self.T}"
        s += f"\n\tdeg: {self.deg}"
        s += f"\n\tnum_funs: {self.num_funs}"
        s += f"\n\tnum_bubb_funs: {self.num_bubb_funs}"
        s += f"\n\tnum_vert_funs: {self.num_vert_funs}"
        s += f"\n\tnum_edge_funs: {self.num_edge_funs}"
        return s

    # SETTERS ################################################################

    def set_mesh(self, T: PlanarMesh) -> None:
        """
        Set the mesh T.

        Parameters
        ----------
        T : PlanarMesh
            Planar mesh
        """
        if not isinstance(T, PlanarMesh):
            raise TypeError("T must be a PlanarMesh")
        self.T = T

    def set_deg(self, deg: int) -> None:
        """
        Set the degree of the polynomial space.

        Parameters
        ----------
        deg : int
            Degree of polynomial space
        """
        if not isinstance(deg, int):
            raise TypeError("deg must be an integer")
        if deg < 1:
            raise ValueError("deg must be a positive integer")
        self.deg = deg

    # BUILD LOCAL FUNCTION SPACES ############################################

    def build_edge_spaces(self, verbose: bool = True) -> None:
        """
        Build the Edge spaces for each edge in the mesh.

        Parameters
        ----------
        verbose : bool, optional
            If True, print progress bars, by default True
        """
        self.edge_spaces = []
        if verbose:
            print("Building Edge spaces...")
            for e in tqdm(self.T.edges):
                e.parameterize(quad_dict=self.quad_dict)
                self.edge_spaces.append(EdgeSpace(e, self.deg))
                e.deparameterize()
        else:
            for e in self.T.edges:
                e.parameterize(quad_dict=self.quad_dict)
                self.edge_spaces.append(EdgeSpace(e, self.deg))
                e.deparameterize()

    def build_local_function_space(
        self,
        cell_idx: int,
        verbose: bool = True,
        compute_interior_values: bool = True,
        compute_interior_gradient: bool = True,
    ) -> LocalPoissonSpace:
        """
        Build the local function space V_p(K) for a MeshCell K.

        Parameters
        ----------
        cell_idx : int
            Index of the cell K in the mesh
        verbose : bool, optional
            If True, print progress bars, by default True
        compute_interior_values : bool, optional
            If True, compute the values of interior functions, by default True
        """
        abs_cell_idx = self.T.get_abs_cell_idx(cell_idx)
        K = self.T.get_cells(cell_idx)
        K.parameterize(quad_dict=self.quad_dict)
        edge_spaces = []
        for e in K.get_edges():
            b = self.edge_spaces[e.idx]
            edge_spaces.append(b)
        V_K = LocalPoissonSpace(
            K,
            edge_spaces,
            self.deg,
            verbose=verbose,
            compute_interior_values=compute_interior_values,
            compute_interior_gradient=compute_interior_gradient,
        )
        for v in V_K.get_basis():
            glob_idx = self.get_global_idx(v.key, abs_cell_idx)
            v.key.set_glob_idx(glob_idx)
            self.cell_dofs[abs_cell_idx].append(v.key)
            v.key.is_on_boundary = self.fun_is_on_boundary(v.key)
        return V_K

    # COUNT FUNCTIONS ########################################################

    def compute_dimension(self) -> None:
        """
        Compute the dimension of the function space.

        The following attributes are computed:
        - num_funs: total number of functions
        - num_vert_funs: number of vertex functions
        - num_edge_funs: number of edge functions
        - num_bubb_funs: number of bubble functions
        """
        self.compute_num_vert_funs()
        self.compute_num_edge_funs()
        self.compute_num_bubb_funs()
        self.num_funs = (
            self.num_vert_funs + self.num_edge_funs + self.num_bubb_funs
        )

    def compute_num_vert_funs(self) -> None:
        """
        Compute the number of vertex functions.

        The following attributes are computed:
        - num_vert_funs: number of vertex functions
        """
        self.num_vert_funs = self.T.num_verts

    def compute_num_edge_funs(self) -> None:
        """
        Compute the number of edge functions.

        The following attributes are computed:
        - num_edge_funs: number of edge functions
        """
        self.num_edge_funs = 0
        for b in self.edge_spaces:
            self.num_edge_funs += b.num_edge_funs

    def compute_num_bubb_funs(self) -> None:
        """
        Compute the number of bubble functions.

        The following attributes are computed:
        - num_bubb_funs: number of bubble functions
        """
        num_bubb = (self.deg * (self.deg - 1)) // 2
        self.num_bubb_funs = self.T.num_cells * num_bubb

    def compute_edge_fun_cumsum(self) -> None:
        """
        Compute the cumulative sum of the number of edge functions.

        The following attribute is computed:
        - edge_fun_cumsum: cumulative sum of the number of edge functions
        """
        self.edge_fun_cumsum = [0]
        for b in self.edge_spaces:
            self.edge_fun_cumsum.append(
                self.edge_fun_cumsum[-1] + b.num_edge_funs
            )

    # LOCAL-TO-GLOBAL MAPPING ################################################

    def get_global_idx(self, key: GlobalKey, abs_cell_idx: int) -> int:
        """
        Return the global index of a local function.

        Parameters
        ----------
        key : GlobalKey
            Local key of the function
        abs_cell_idx : int
            Absolute cell index

        Returns
        -------
        int
            Global index of the function
        """
        BUBB_START_IDX = 0
        VERT_START_IDX = self.num_bubb_funs
        EDGE_START_IDX = self.num_bubb_funs + self.num_vert_funs
        NUM_BUBB_CELL = (self.deg * (self.deg - 1)) // 2
        if key.fun_type == "bubb":
            idx = BUBB_START_IDX
            idx += NUM_BUBB_CELL * abs_cell_idx
            idx += key.bubb_space_idx
            if not self.is_in_range(idx, BUBB_START_IDX, VERT_START_IDX):
                raise IndexError("bubble function index out of range")
        if key.fun_type == "vert":
            idx = VERT_START_IDX
            idx += self.T.vert_idx_list.index(key.vert_idx)
            if not self.is_in_range(idx, VERT_START_IDX, EDGE_START_IDX):
                raise IndexError("vertex function index out of range")
        if key.fun_type == "edge":
            idx = self.num_bubb_funs + self.num_vert_funs
            idx += self.edge_fun_cumsum[key.edge_idx]
            idx += key.edge_space_idx
            if not self.is_in_range(idx, EDGE_START_IDX, self.num_funs):
                raise IndexError("Edge function index out of range")
        return idx

    def is_in_range(self, idx: int, lo: int, hi: int) -> bool:
        """
        Return True if idx is in the range [lo, hi).

        Parameters
        ----------
        idx : int
            Index to check
        lo : int
            Lower bound
        hi : int
            Upper bound

        Returns
        -------
        bool
            True if idx is in the range [lo, hi)
        """
        return lo <= idx < hi

    def fun_is_on_boundary(self, key: GlobalKey) -> bool:
        """
        Return True if a global function is on the boundary.

        Parameters
        ----------
        key : GlobalKey
            Global key of the function
        """
        if key.fun_type == "bubb":
            return False
        if key.fun_type == "vert":
            return self.T.vert_is_on_boundary(key.vert_idx)
        if key.fun_type == "edge":
            return self.T.edge_is_on_boundary(key.edge_idx)
        raise ValueError("Invalid function type")
