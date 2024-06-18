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
    mesh : PlanarMesh
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

    mesh: PlanarMesh
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
        mesh: PlanarMesh,
        deg: int,
        quad_dict: QuadDict,
        verbose: bool = True,
    ) -> None:
        """
        Initialize a GlobalFunctionSpace object.

        Parameters
        ----------
        mesh : PlanarMesh
            Planar mesh
        deg : int
            Degree of polynomial space
        quad_dict : dict[str, Quad]
            Dictionary of quadrature rules
        verbose : bool, optional
            If True, print progress bars, by default True
        """
        self.quad_dict = quad_dict
        self.cell_dofs = [[] for _ in range(mesh.num_cells)]
        self.set_mesh(mesh)
        self.set_deg(deg)
        self.build_edge_spaces(verbose=verbose)
        self._compute_dimension()
        self._compute_edge_fun_cumsum()
        if verbose:
            print(self)

    def __str__(self) -> str:
        """Return a string representation of the global function space."""
        s = "GlobalFunctionSpace:"
        s += f"\n\tmesh: {self.mesh}"
        s += f"\n\tdeg: {self.deg}"
        s += f"\n\tnum_funs: {self.num_funs}"
        s += f"\n\tnum_bubb_funs: {self.num_bubb_funs}"
        s += f"\n\tnum_vert_funs: {self.num_vert_funs}"
        s += f"\n\tnum_edge_funs: {self.num_edge_funs}"
        return s

    # SETTERS ################################################################

    def set_mesh(self, mesh: PlanarMesh) -> None:
        """
        Set the mesh.

        Parameters
        ----------
        mesh : PlanarMesh
            Planar mesh
        """
        if not isinstance(mesh, PlanarMesh):
            raise TypeError("'mesh' must be a PlanarMesh")
        self.mesh = mesh

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
        if deg > 3:
            print(
                "Warning: deg > 3 is unstable and may result in a poorly "
                + "conditioned system"
            )
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
            for e in tqdm(self.mesh.edges):
                e.parameterize(quad_dict=self.quad_dict)
                self.edge_spaces.append(EdgeSpace(e, self.deg))
                e.deparameterize()
        else:
            for e in self.mesh.edges:
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
        abs_cell_idx = self.mesh.get_abs_cell_idx(cell_idx)
        mesh_cell = self.mesh.get_cell(cell_idx)
        mesh_cell.parameterize(
            quad_dict=self.quad_dict,
            compute_interior_points=compute_interior_values
            or compute_interior_gradient,
        )
        edge_spaces = []
        for e in mesh_cell.get_edges():
            b = self.edge_spaces[e.idx]
            edge_spaces.append(b)
        local_fun_sp = LocalPoissonSpace(
            mesh_cell,
            edge_spaces,
            self.deg,
            verbose=verbose,
            compute_interior_values=compute_interior_values,
            compute_interior_gradient=compute_interior_gradient,
        )
        for v in local_fun_sp.get_basis():
            glob_idx = self.get_global_idx(v.key, abs_cell_idx)
            v.key.set_glob_idx(glob_idx)
            self.cell_dofs[abs_cell_idx].append(v.key)
            v.key.is_on_boundary = self.fun_is_on_boundary(v.key)
        return local_fun_sp

    # COUNT FUNCTIONS ########################################################

    def _compute_dimension(self) -> None:
        self._compute_num_vert_funs()
        self._compute_num_edge_funs()
        self._compute_num_bubb_funs()
        self.num_funs = (
            self.num_vert_funs + self.num_edge_funs + self.num_bubb_funs
        )

    def _compute_num_vert_funs(self) -> None:
        self.num_vert_funs = self.mesh.num_verts

    def _compute_num_edge_funs(self) -> None:
        self.num_edge_funs = 0
        for b in self.edge_spaces:
            self.num_edge_funs += b.num_edge_funs

    def _compute_num_bubb_funs(self) -> None:
        num_bubb = (self.deg * (self.deg - 1)) // 2
        self.num_bubb_funs = self.mesh.num_cells * num_bubb

    def _compute_edge_fun_cumsum(self) -> None:
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
        bubb_start_idx = 0
        vert_start_idx = self.num_bubb_funs
        edge_start_idx = self.num_bubb_funs + self.num_vert_funs
        num_bubb_cell = (self.deg * (self.deg - 1)) // 2
        if key.fun_type == "bubb":
            idx = bubb_start_idx
            idx += num_bubb_cell * abs_cell_idx
            idx += key.bubb_space_idx
            if not self._is_in_range(idx, bubb_start_idx, vert_start_idx):
                raise IndexError("bubble function index out of range")
        if key.fun_type == "vert":
            idx = vert_start_idx
            idx += self.mesh.vert_idx_list.index(key.vert_idx)
            if not self._is_in_range(idx, vert_start_idx, edge_start_idx):
                raise IndexError("vertex function index out of range")
        if key.fun_type == "edge":
            idx = self.num_bubb_funs + self.num_vert_funs
            idx += self.edge_fun_cumsum[key.edge_idx]
            idx += key.edge_space_idx
            if not self._is_in_range(idx, edge_start_idx, self.num_funs):
                raise IndexError("edge function index out of range")
        return idx

    def _is_in_range(self, idx: int, lo: int, hi: int) -> bool:
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
            return self.mesh.vert_is_on_boundary(key.vert_idx)
        if key.fun_type == "edge":
            return self.mesh.edge_is_on_boundary(key.edge_idx)
        raise ValueError("Invalid function type")
