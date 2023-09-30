"""
global_function_space.py
========================

Module containing the global_function_space class, which is used to represent a
global function space.
"""

from tqdm import tqdm

from ..locfun.edge_space import edge_space
from ..locfun.locfunsp import locfunspace
from ..mesh.planar_mesh import planar_mesh
from ..mesh.quad import quad
from .globkey import global_key


class global_function_space:
    """
    Represents a global function space V_p(T) on a planar mesh T of degree p.
    """

    T: planar_mesh
    deg: int
    quad_dict: dict[str, quad]
    num_funs: int
    num_vert_funs: int
    num_edge_funs: int
    num_bubb_funs: int
    edge_spaces: list[edge_space]
    edge_fun_cumsum: list[int]
    cell_dofs: list[list[global_key]]

    def __init__(
        self,
        T: planar_mesh,
        deg: int,
        quad_dict: dict[str, quad],
        verbose: bool = True,
    ) -> None:
        """
        Constructor for global_function_space class.

        Parameters
        ----------
        T : planar_mesh
            Planar mesh
        deg : int
            Degree of polynomial space
        quad_dict : dict[str, quad]
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
        """
        Returns a string representation of the global function space.
        """
        s = "global_function_space:"
        s += f"\n\tT: {self.T}"
        s += f"\n\tdeg: {self.deg}"
        s += f"\n\tnum_funs: {self.num_funs}"
        s += f"\n\tnum_bubb_funs: {self.num_bubb_funs}"
        s += f"\n\tnum_vert_funs: {self.num_vert_funs}"
        s += f"\n\tnum_edge_funs: {self.num_edge_funs}"
        return s

    # SETTERS ################################################################

    def set_mesh(self, T: planar_mesh) -> None:
        """
        Sets the mesh T.
        """
        if not isinstance(T, planar_mesh):
            raise TypeError("T must be a planar_mesh")
        self.T = T

    def set_deg(self, deg: int) -> None:
        """
        Sets the degree of the polynomial space.
        """
        if not isinstance(deg, int):
            raise TypeError("deg must be an integer")
        if deg < 1:
            raise ValueError("deg must be a positive integer")
        self.deg = deg

    # BUILD LOCAL FUNCTION SPACES ############################################

    def build_edge_spaces(self, verbose: bool = True) -> None:
        """
        Builds the edge spaces for each edge in the mesh.
        """
        self.edge_spaces = []
        if verbose:
            print("Building edge spaces...")
            for e in tqdm(self.T.edges):
                e.parameterize(quad_dict=self.quad_dict)
                self.edge_spaces.append(edge_space(e, self.deg))
                e.deparameterize()
        else:
            for e in self.T.edges:
                e.parameterize(quad_dict=self.quad_dict)
                self.edge_spaces.append(edge_space(e, self.deg))
                e.deparameterize()

    def build_local_function_space(
        self,
        cell_idx: int,
        verbose: bool = True,
    ) -> locfunspace:
        """
        Builds the local function space V_p(K) for a cell K.
        """
        abs_cell_idx = self.T.get_abs_cell_idx(cell_idx)
        K = self.T.get_cell(cell_idx)
        K.parameterize(quad_dict=self.quad_dict)
        edge_spaces = []
        for e in K.get_edges():
            b = self.edge_spaces[e.id]
            edge_spaces.append(b)
        V_K = locfunspace(K, edge_spaces, self.deg, verbose=verbose)
        for v in V_K.get_basis():
            glob_idx = self.get_global_idx(v.id, abs_cell_idx)
            v.id.set_glob_idx(glob_idx)
            self.cell_dofs[abs_cell_idx].append(v.id)
            v.id.is_on_boundary = self.fun_is_on_boundary(v.id)
        return V_K

    # COUNT FUNCTIONS ########################################################

    def compute_dimension(self) -> None:
        """
        Computes the dimension of the function space.
        """
        self.compute_num_vert_funs()
        self.compute_num_edge_funs()
        self.compute_num_bubb_funs()
        self.num_funs = (
            self.num_vert_funs + self.num_edge_funs + self.num_bubb_funs
        )

    def compute_num_vert_funs(self) -> None:
        """
        Computes the number of vertex functions.
        """
        self.num_vert_funs = self.T.num_verts

    def compute_num_edge_funs(self) -> None:
        """
        Computes the number of edge functions.
        """
        self.num_edge_funs = 0
        for b in self.edge_spaces:
            self.num_edge_funs += b.num_edge_funs

    def compute_num_bubb_funs(self) -> None:
        """
        Computes the number of bubble functions.
        """
        num_bubb = (self.deg * (self.deg - 1)) // 2
        self.num_bubb_funs = self.T.num_cells * num_bubb

    def compute_edge_fun_cumsum(self) -> None:
        """
        Computes the cumulative sum of the number of edge functions.
        """
        self.edge_fun_cumsum = [0]
        for b in self.edge_spaces:
            self.edge_fun_cumsum.append(
                self.edge_fun_cumsum[-1] + b.num_edge_funs
            )

    # LOCAL-TO-GLOBAL MAPPING ################################################

    def get_global_idx(self, id: global_key, abs_cell_idx: int) -> int:
        """
        Returns the global index of a local function.
        """
        BUBB_START_IDX = 0
        VERT_START_IDX = self.num_bubb_funs
        EDGE_START_IDX = self.num_bubb_funs + self.num_vert_funs
        NUM_BUBB_CELL = (self.deg * (self.deg - 1)) // 2
        if id.fun_type == "bubb":
            idx = BUBB_START_IDX
            idx += NUM_BUBB_CELL * abs_cell_idx
            idx += id.bubb_space_idx
            if not self.is_in_range(idx, BUBB_START_IDX, VERT_START_IDX):
                raise IndexError("bubble function index out of range")
        if id.fun_type == "vert":
            idx = VERT_START_IDX
            idx += self.T.vert_idx_list.index(id.vert_idx)
            if not self.is_in_range(idx, VERT_START_IDX, EDGE_START_IDX):
                raise IndexError("vertex function index out of range")
        if id.fun_type == "edge":
            idx = self.num_bubb_funs + self.num_vert_funs
            idx += self.edge_fun_cumsum[id.edge_idx]
            idx += id.edge_space_idx
            if not self.is_in_range(idx, EDGE_START_IDX, self.num_funs):
                raise IndexError("edge function index out of range")
        return idx

    def is_in_range(self, idx: int, lo: int, hi: int) -> bool:
        """
        Returns True if idx is in the range [lo, hi).
        """
        return lo <= idx < hi

    def fun_is_on_boundary(self, id: global_key) -> bool:
        """
        Returns True if a global function is on the boundary.
        """
        if id.fun_type == "bubb":
            return False
        if id.fun_type == "vert":
            return self.T.vert_is_on_boundary(id.vert_idx)
        if id.fun_type == "edge":
            return self.T.edge_is_on_boundary(id.edge_idx)
        raise ValueError("Invalid function type")
