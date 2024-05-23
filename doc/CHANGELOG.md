# Changelog


## [Unreleased]
### Examples
- [x] remove import management boilerplate from examples
- [x] change example naming convention
- [ ] partition examples into tutorials and examples
    - [ ] tutorials belong to the user guide in `doc/tutorials/`: describe basic functionality
    - [ ] examples belong to the `examples/` directory: examples used (or will be used) in papers, advanced functionality, applications
- [x] update examples to use new initialization method for `LocalFunction` objects
- [ ] eigenvalue problem
### Documentation
- [x] change: installation instructions for developers to build the package locally with `pip install -e .`
- [x] add: tutorials
- [x] change: update docstrings to use NumPy documentation format
    - [x] `locfun`
    - [x] `mesh` (except for `edgelib`)
    - [x] `plot`
    - [x] `solver`
    - [x] `util`
- [x] change: MkDocs theme to `material`
- [x] add: site logo
- [ ] update `doc/logo/pacman.svg`
- [ ] tutorials
    - [ ] Guide to tutorials
    - [ ] Meshes
        - [ ] 1.1 Vertices and edges (taken from Example 0.1)
        - [ ] 1.2 Mesh cells and planar meshes (taken from Example 0.1)
    - [ ] Local function spaces
        - [ ] 2.2 Traces (taken from Example 0.2)
        - [ ] 2.5 Heavy sampling of edges (taken from Example 1.5)
- [ ] run tutorial notebooks when building documentation
- [ ] link to notebook source on tutorial pages
- [ ] link to tutorial pages from README
- [ ] move installation instructions using `git` from CONTRIBUTING to INSTALLATION
### Maintenance
- [x] `DirichletTrace` improvements:
    - [x] add: weighted normal derivative
    - [x] add: weighted tangential derivative
- [x] `LocalFunction` improvements:
    - [x] BREAKING CHANGE: replace the handling of traces with `DirichletTrace`
    - [x] DEPRECATED: use `LocalPoissonFunction` instead
- [x] add: `LocalPoissonFunction` class, built from `LocalHarmonic` and `LocalPolynomial` objects
- [x] `NystromSolver` improvements:
    - [x] handle logarithmic functions as instances of `DirichletTrace` class
    - [x] move `jacobi_preconditioner()` static method to a separate module
    - [x] `antilap_strategy` option to precompute "normal indicator functions" biharmonic function computation speedup
- [x] `locfun` module improvements:
    - [x] use `DirichletTrace` objects for traces in `antilap` module
    - [x] move contents of `antilap` and `d2n` subpackages to `locfun` root
    - [x] remove `log_terms` and `log_antilap` modules
    - [x] deprecate `PiecewisePolynomial` class
- [x] `Polynomial` improvements:
  - [x] deprecate `eval()` method, use `__call__()` method instead
- [ ] replace `GlobalFunctionSpace` with `GlobalPoissionSpace`:
    - [ ] deprecate `GlobalFunctionSpace`
    - [ ] build from a collection of `GlobalFunction` objects, just like `LocalFunctionSpace` is built from a collection of `LocalFunction` objects
    - [ ] precompute `LocalFunctionSpace` objects
- [ ] `MeshPlot` improvements:
    - [ ] change: initialize with either a list of `Edge` objects, a `MeshCell`, or a `PlanarMesh`
    - [ ] add: plot interior points with `draw_interior_points()` method
    - [ ] add: plot interior point triangulation with `draw_interior_triangulation()` method
- [ ] `LocalFunctionPlot` improvements:
    - [ ] treat plot of local function as a special case of global function with a single cell
- [ ] `GlobalFunctionPlot` improvements:
    - [ ] interior value interpolation for points close to the boundary
    - [ ] consolidate the multiple draw methods into one, optional argument to specify what to draw (values, gradient, etc.)
### Bug Fixes
- [x] fix: missing logo and favicon in MkDocs build
- [ ] fix: broken links in examples
- [ ] fix: broken links in tutorials
- [ ] fix: `align*` blocks not rendering in tutorials when building documentation
- [ ] fix: `DirichletTrace` does not recompute traces when function changes

## [v0.4.5] - 2024 Apr 24
### Maintenance
- [x] change dependency: Python 3.9 (was 3.11)
- [x] change dependency: `scipy` 1.12 (was 1.11)
- [x] change: use `rtol` keyword argument in `scipy.sparse.linalg.gmres` to `tol`
- [x] fix: missing type annotations for tests
- [x] fix: use `Union` for type hints rather than ` | ` operator (to support Python 3.9)


## [v0.4.4] - 2024 Apr 07
### Documentation
- [x] change: `README.md`
    - [x] add: installation instructions with `pip`
    - [x] add: usage instructions via `readthedocs` link
- [x] change: consolidate `TODO.md` and `WISHLIST.md` into `ROADMAP.md`
- [x] add: documentation with MkDocs
    - [x] add: `.readthedocs.yml` for ReadTheDocs configuration
    - [x] add: `mkdocs.yml` for MkDocs configuration
    - [x] add: `doc/requirements.txt` for MkDocs dependencies
    - [x] add: `doc/index.md` for home page
    - [x] add: quickstart guide
        - [x] add: installation instructions
    - [x] add: user guide
        - [x] mesh:
            - [x] `Edge` class
            - [x] `ClosedContour` class
            - [x] `MeshCell` class
            - [x] `PlanarMesh` class
            - [x] `Quad` class
            - [x] `QuadDict` class
            - [x] `Vert` class
            - [x] `get_quad_dict()` function
            - [x] `meshlib` module
            - [x] `mesh_builder` function
            - [x] `split_edge` function
        - [x] local function spaces:
            - [x] `Polynomial` class
            - [x] `DirichletTrace` class
            - [x] `NystromSolver` class
            - [x] `LocalFunction` class
            - [x] `LocalFunctionSpace` class
        - [x] FEM solver:
            - [x] `BilinearForm` class
            - [x] `GlobalFunctionSpace` class
            - [x] `Solver` class
        - [x] plotting:
            - [x] `GlobalFunctionPlot` class
            - [x] `LocalFunctionPlot` class
            - [x] `MeshPlot` class
            - [x] `TracePlot` class
    -  [x] add: developer guide
        - [x] roadmap
        - [x] contributing guide
        - [x] changelog
### Package Management
- [x] add: `poetry` configuration for package management
    - [x] change: `pyproject.toml` to use `poetry`
    - [x] add: `poetry.lock`


## [v0.4.3] - 2024 Mar 19
### Examples
- [x] modify: `ex1a` to use `DirichletTrace`
- [x] modify: `ex1e` to use `DirichletTrace`
- [x] rename: `ex0` to `ex0a`
- [x] add: `ex0b` to demonstrate how to construct a `LocalFunction` with a `DirichletTrace`
### Features
- [x] add: `DirichletTrace` class for handling the traces of `LocalFunction`s
    - construct an arbitrary trace, or a polynomial trace in the style of `LocalFunctionSpace`
    - will someday replace the default handling of traces in `LocalFunction` class
- [x] modify: `Polynomial` objects are now callable
- [x] add: splitting an edge into multiple edges without recursive subdivision
- [x] add: optional argument to `TracePlot` initializer to specify the maximum number of ticks on the horizontal axis
### Maintenance
- [x] add: debug option to `NystromSolver` to show condition number
- [x] preconditioning for `NystromSolver`
- [x] use `numba` to speed up `NystromSolver` matrix assembly with just-in-time compilation
- [x] pass `DirichletTrace` object to the `TracePlot` constructor
### Bug fixes
- [x] fix: too many tick marks on `TracePlot`


## [v0.4.2] - 2024 Feb 29
### Documentation
- [x] add a contributing guide
### Features
- [x] make minimum distance to boundary for interior points of a `MeshCell` adjustable with `set_interior_point_tolerance()` method
- [x] make computation of interior gradients optional
- [x] add methods to `MeshCell` to get unit tangent and unit normal vectors, and the derivative norm
### Maintenance
- [x] vectorize interior value computation
### Tests
- [x] add ghost cell to mesh cell testing library
- [x] add test for interior points
### Bug Fixes
- [x] Plots of global solution corrupted: fix by not recording edge flips to transformation diary


## [v0.4.1] - 2024 Feb 25
### Examples
- [x] add cubic spline interpolation example to `ex0`
### Maintenance
- [x] remove trigonometric interpolation
- [x] remove trigonometric interpolation tests
- [x] format `edgelib/spline`


## [v0.4.0] - 2024 Feb 25
### Documentation
- [x] update README, extend description, add references
### Features
- [x] add ability to define an edge parameterization using a cubic spline to interpolate points (thanks, Zack!)
- [x] add edge splitting
- [x] add transformation diary to `Edge` class
- [x] add global stiffness and mass matrices to `solver` class
- [x] add option to turn off axes in plots
- [x] add colormap option to contour plot methods
- [x] add trigonometric interpolation for cell boundary traces
### Examples
- [x] add space-filling curve example (thanks, Zack!)
- [x] add `ex1e` to demonstrate heavy sampling of edges via edge splitting
### Maintenance
- [x] make colorbar optional for `GlobalFunctionPlot` draw method
- [x] make coefficients optional in `GlobalFunctionPlot` init method
- [x] add warning for `Quad` class when `n > 128 * interp`
- [x] use kwargs for plotting options
- [x] add `PiecewisePolynomial` class to init file
- [x] create `QuadDict` object to standardize quadrature collections
### Tests
- [x] add test for trigonometric interpolation
### Bug Fixes
- [x] close figure in `draw` methods for plotting classes
- [x] fix `show_plot=False` option not working in `draw` methods for plotting classes
- [x] make directory to save plots if it doesn't exist
- [x] fix `LocalFunctionPlot` saving blank files
- [x] fix Martensen quadrature for large values of n
- [x] raise exception when Nystrom solver encounters non-numeric values


## [v0.3.8] - 2023 Oct 27
### Features
- [x] add `TracePlot` class
- [x] add `MeshPlot` class
- [x] add `GlobalFunctionPlot` class
- [x] add `LocalFunctionPlot` class
- [x] add `get_quad_dict()` function
- [x] make `edgespaces` an optional parameter in `LocalFunctionSpace` init method
### Examples
- [x] `examples/ex1d-hat-tile.ipynb`: add example of hat tile local basis functions
### Tests
- [x] deprecate `unittest` in favor of `pytest`
- [x] add `test_edge_space`
- [x] add `test_solver`
### Maintenance
- [x] use enumerate to replace `range(len(...))` loops
- [x] rectify nested min/max statements
- [x] `locfun.locfunsp`: make interior value calculation optional
- [x] use `functools.partial` to pass logarithmic terms to integrators
- [x] `locfun.poly.poly`: use list of tuples for polynomial initialization
- [x] clean up integration methods
- [x] add safety check to `add_edge()` method in `PlanarMesh`
- [x] `mesh.edgelib.teardrop`: pass `alpha` as keyword argument
- [x] `solver.solver`:
    - [x] move color printing to separate module
    - [x] move plotting functions to separate module
- [x] add init file for `util` subpackage
- [x] rename classes to use CapWords convention
- [x] fix invalid names introduced by class renaming
- [x] fix type hints in tests
### Bug fixes
- [x] fix colorbar position in contour plots


## [v0.3.7-alpha] - 2023 Oct 02
- [x] change exceptions to specific error types
- [x] document TODO comments in `doc/TODO.md`
- [x] modify `CHANGELOG.md`, `TODO.md`, `WISHLIST.md` to reflect semantic versioning
    - future versions will use `git` branches to isolate development
    - git tags and GitHub releases will be used to track versions
    - small commits get a descriptive message
    - branch merges get a release number


## [v0.3.6] - 2023 Sep 30
- rename `id` variables to either `idx` or `key` to avoid shadowing built-ins


## [v0.3.5] - 2023 Sep 30
- write docstrings
    - modules
    - classes
    - methods
    - functions
- add `__init__.py` to `solver` subpackage


## [v0.3.4] - 2023 Sep 20
- use generators where appropriate
- fix imports
- use f-strings
- fix spelling errors
- use `enumerate` in loops where appropriate
- fix superfluous `return` statements
- fix unnecessary `else` statements
- change underscore methods in `edgelib` to capitalized
    - update `edge` class to reflect change
    - update `ex0` example to reflect change
- add type hints
    - functions
    - methods
    - classes
- fix other `pylint` and `mypy` errors
- update `pyproject.toml`


## [v0.3.3] - 2023 Sep 20
- rename `setup.cgf` to `.flake8` (until `flake8` supports `pyproject.toml`)
- update `devtools/convert_examples.sh` to support being executed from any directory


## [v0.3.2] - 2023 Sep 19
- added support for `pytest`
- added `pytest` and dependencies to `requirements-dev.txt`
- added `pytest` parameters to `pyproject.toml`


## [v0.3.1] - 2023 Sep 19
- Reformatted with `black` and `isort`
- Linted with `flake8` and `pylint` (ongoing)
- Type-checked with `mypy` (ongoing)
- Added `requirements.txt` and `requirements-dev.txt`
- Added `CHECKLIST.md`
- Added `setup.cnf`, `pyproject.toml`

## [v0.3.0] - 2023 Aug 07
### Major changes
* Added local function spaces (`locfunspace`) to `locfun/`
    * Class containing a basis of the local Poisson space $V_p(K)$
* Added edge spaces (`edgespace`) to `locfun/`
    * Added support for high order (including $p > 3$, unstable for some edges)
    * Added barycentric coordinates and Legendre polynomials to `poly/`
    * High order edge spaces obtained by traces of integrated Legendre polynomials
* Overhaul of `locfun` traces
    * Local function traces are by default considered a list of `polynomial`s, one for each edge
  * This list of `polynomial`s is stored in `trace_polys` attribute
  * Trace values are computed only as needed
  * Trace values can be set manually as before
  * The flag `has_poly_trace` can be set to `False` if working functions with functions that do not have a trace that is the trace of a polynomial on each edge
  * `locfun` objects are now initialized with the Laplacian polynomial, the list of polynomial traces, and an option
* Overhaul of Nystrom solver
    * Replaced `nystrom` module with `nystrom_solver` class
    * Overhead of constructing the Nystrom matrix is now consolidated into the constructor
  * Harmonic conjugate computations are now handled as method of `nystrom_solver`
* Added `vert` class
    * Vertices of a mesh
* Overhaul of `edge` class
    * Added topological properties needed for mesh construction
    * Parameterization values computed and stored only as needed
    * Removed dependence on `copy` package
    * Added `integrate_over_edge` method
    * Added `global_orientation` attribute
* Overhaul of `cell` class
    * Added topological properties needed for mesh construction
    * Edge list replaced with `closed_contour` list
* Added `PlanarMesh` class
    * Initialized as a collection of `edge` and `vert` objects
    * `cell` objects are constructed in situ using topological information from `edge` objects
* Added `solver` class
    * Handles all aspects of solving a PDE on a mesh
    * Initialized with a `PlanarMesh` object and a `bilinear_form` object
    * `solve` method solves the PDE on the mesh
    * `plot` method plots the solution
* Added `bilinear_form` class
    * Stores info about the PDE to be solved
    * Also stores the right-hand side of the PDE as a polynomial
* Added `global_function_space` class to manage global function space
* Added `global_key` class to manage continuity across cells
### Minor changes:
* Moved `quad` module to `mesh/` from `mesh/quad/`
* Added `polynomial` functionality
    * Added division of a polynomial by a scalar
    * Added powers of polynomials
    * Added polynomial composition
* Changed directory management in Jupyter notebook examples
* Added `devtools/` directory
* Added `doc/` directory
* Added `.py` versions of examples
    * Thanks to `nbconvert` for making this easy!
* Added `ex2-pacman-fem` example
* Updated tests and examples

## [v0.2.5] - 2023 May 01
* Moved examples to a dedicated `examples` directory
    * Added "Examples used in publications" to `README`
* Introduced API for more convenient user experience
    * Users call `pf.thing` rather than `pf.foo.bar.baz.thing`
    * Updated all examples to reflect API change
    * Restructured `puncturedfem` directory
* Polynomial overhaul
    * Changed initialization method to something sensible
    * `set()` method behaves as a re-initializer
    * Added support for polynomial-scalar addition for `+`,`+=`,`-`,`-=` operators
    * Added support for `*=` operator
    * Moved `integrate_over_cell()` method to `locfun.int_poly`
* Unit tests
    * Relocated `test` to parent directory
    * Polynomials: `test_poly.py`
    * Local functions: `test_locfun.py`

## [v0.2.4] - 2023 Apr 17
* Added `intval` module to `locfun` for computing interior values
* Added `get_conjugable_part()` method to `locfun` that returns the trace of $\psi$
* Renamed `is_in_interior()` method for `contour` to `is_in_interior_contour()`
* Added `is_in_interior_cell()` method to `cell`
* Added interior value demo to Punctured Square, Pac-Man, and Ghost examples

## [v0.2.3] - 2023 Mar 06
* Added **ex1b-pacman.ipynb**
* Added **ex1c-ghost.ipynb**
* Updated **ex1a-square-hole.ipynb** with more accurate reference values

## [v0.2.2] - 2023 Mar 01
* Added `compute_h1()` and `compute_l2()` methods to `locfun`
* Added `integrate_over_cell()` method to `polynomial`
* Added methods to `contour` and `cell` to integrate over the boundary without multiplying by the norm of the derivative (i.e. 'preweighted' integrands)
* Monomials now default to zero monomial
* Fixed evaluation of zero `monomial` and zero `polynomial` to return same size as input
* Removed `ext_pt` field from `cell`
* Added `id` field to `contour`
* Added `get_integrator()` method to `contour` (`cell` inherits)
* Added `neumann` module to `nystrom` for solving Neumann problem
    * Modified `harmconj` module accordingly
* Completely overhauled anti-Laplacian calculation for punctured cells
* **ex1**: "Square with circular hole" updated

## [v0.2.1] - 2023 Feb 25
* Added **/poly** subpackage
    * `monomial` and `polynomial` objects
* Added **locfun/** subpackage
    * `locfun` object holds all data for local function $v\in V_p(K)$
* Added `ext_pt` field to `cell` object, which is an exterior point such that centering the origin at `ext_pt` placing the cell strictly in the first quadrant

## [v0.2.0] - 2023 Feb 16
* MATLAB code rewritten in Python to increase accessibility
* Examples presented with Jupyter Notebook
    - **ex0-mesh-building**:
      defining edges, cells, and meshes
    - **ex1-inner-prod**:
      compute $H^1$ and $L^2$ (semi-)inner products on punctured cell
* New subpackages/modules
    - **quad**: trapezoid, Kress, and Martensen quadrature objects
    - **mesh**: mesh construction tools
        - **edge**: parameterization of an edge
        - **contour**: collection of edges forming a simple closed contour
        - **cell**: mesh cell (subclass of contour)
    - **plot**: functions for
        - plot edges, contours, cell boundaries
        - trace of a function along a collection of edges
    - **nystrom**: Nystr$\text{\"o}$m method for solving integral equations
      - includes single and double layer operators
      - block system support
    - **d2n**: Dirichlet-to-Neumann map for harmonic functions
      - computation of harmonic conjugate
      - FFT (anti-)derivatives
    - **antilap**: tools to compute anti-Laplacians of harmonic functions
* Added unit tests to **puncturedfem/test/**
  - **test_harmconj** for harmonic conjugates
  - **test_fft_deriv** for FFT differentiation

## [v0.1.0] - 2022 Aug 02
Only a simple diffusion operator (the Laplacian) is currently supported.
Dirichlet and mixed Dirichlet-Neumann boundary conditions are available,
but are assumed to be homogeneous. Used to run a simple "pegboard" example.
