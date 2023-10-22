# Punctured FEM: Change Log


## [yyyy mm dd] v0.3.8
### Examples
- [ ] `examples/ex1d-hat-tile.ipynb`: add example of hat tile local basis functions
### Tests
- [x] deprecate `unittest` in favor of `pytest`
- [x] add `test_edge_space`
- [x] add `test_solver`
### Maintenance
- [ ] add logging with `logging` module
- [x] use enumerate to replace `range(len(...))` loops
- [x] rectify nested min/max statements
- [ ] `locfun.locfun`: move interior value calculation to separate module
- [x] `locfun.locfunsp`: make interior value calculation optional
- [x] use `functools.partial` to pass logarithmic terms to integrators
- [ ] `locfun.poly.poly`: use enum for `x` and `y` variable references
- [x] `locfun.poly.poly`: use list of tuples for polynomial initialization
- [x] clean up integration methods
- [x] add safety check to `add_edge()` method in `PlanarMesh`
- [x] `mesh.quad`: add `get_quad_dict()` function
- [x] `mesh.edgelib.teardrop`: pass `alpha` as keyword argument
- [x] add `TracePlot` class
- [x] add `MeshPlot` class
- [x] add `GlobalFunctionPlot` class
- [x] `solver.solver`:
  - [x] move color printing to separate module
  - [x] move plotting functions to separate module
- [x] add init file for `util` subpackage
- [x] rename classes to use CapWords convention
- [x] fix invalid names introduced by class renaming
- [x] fix type hints in tests
### Bug fixes
- [x] fix colorbar position in contour plots


## [2023 Oct 02] v0.3.7-alpha
- [x] change exceptions to specific error types
- [x] document TODO comments in `doc/TODO.md`
- [x] modify `CHANGELOG.md`, `TODO.md`, `WISHLIST.md` to reflect semantic versioning
  - future versions will use `git` branches to isolate development
  - git tags and GitHub releases will be used to track versions
  - small commits get a descriptive message
  - branch merges get a release number


## [2023 Sep 30] v0.3.6: Rename id variables
- rename `id` variables to either `idx` or `key` to avoid shadowing built-ins


## [2023 Sep 30] v0.3.5: Docstrings
- write docstrings
  - modules
  - classes
  - methods
  - functions
- add `__init__.py` to `solver` subpackage


## [2023 Sep 20] v0.3.4: Type hints and clean up
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


## [2023 Sep 20] v0.3.3: Flake8 config
- rename `setup.cgf` to `.flake8` (until `flake8` supports `pyproject.toml`)
- update `devtools/convert_examples.sh` to support being executed from any directory


## [2023 Sep 19] v0.3.2: PyTest support
- added support for `pytest`
- added `pytest` and dependencies to `requirements-dev.txt`
- added `pytest` parameters to `pyproject.toml`


## [2023 Sep 19] v0.3.1: Style
- Reformatted with `black` and `isort`
- Linted with `flake8` and `pylint` (ongoing)
- Type-checked with `mypy` (ongoing)
- Added `requirements.txt` and `requirements-dev.txt`
- Added `CHECKLIST.md`
- Added `setup.cnf`, `pyproject.toml`

## [2023 Aug 07] v0.3.0: Meshes
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

## [2023 May 01] v0.2.5: API & polynomial overhaul
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

## [2023 Apr 17] v0.2.4: Interior values
* Added `intval` module to `locfun` for computing interior values
* Added `get_conjugable_part()` method to `locfun` that returns the trace of $\psi$
* Renamed `is_in_interior()` method for `contour` to `is_in_interior_contour()`
* Added `is_in_interior_cell()` method to `cell`
* Added interior value demo to Punctured Square, Pac-Man, and Ghost examples

## [2023 Mar 06] v0.2.3: Pac-Man & Ghost
* Added **ex1b-pacman.ipynb**
* Added **ex1c-ghost.ipynb**
* Updated **ex1a-square-hole.ipynb** with more accurate reference values

## [2023 Mar 01] v0.2.2: Anti-Laplacians
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

## [2023 Feb 25] v0.2.1: Local functions
* Added **/poly** subpackage
  * `monomial` and `polynomial` objects
* Added **locfun/** subpackage
  * `locfun` object holds all data for local function $v\in V_p(K)$
* Added `ext_pt` field to `cell` object, which is an exterior point such that centering the origin at `ext_pt` placing the cell strictly in the first quadrant

## [2023 Feb 16] v0.2.0: Python overhaul
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

## [2022 Aug 02] v0.1.0: Initial commit
Only a simple diffusion operator (the Laplacian) is currently supported.
Dirichlet and mixed Dirichlet-Neumann boundary conditions are available,
but are assumed to be homogeneous. Used to run a simple "pegboard" example.
