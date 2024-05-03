# Roadmap


## Planned Features and Improvements
These are features that are planned for the future, but are not yet implemented.
### Features
- [ ] define an edge parameterization symbolically with `sympy`
- [ ] global boundary conditions
    - [ ] mixed Dirichlet/Neumann boundary conditions
    - [ ] define trace of a global function with `DirichletTrace` class
    - [ ] nonhomogeneous boundary conditions
- [ ] `BilinearForm` improvements
    - [ ] piecewise constant coefficients, load function
    - [ ] advection terms
    - [ ] diffusion terms
### Examples
- [ ] add example of "subdivision refinement"
- [ ] eigenvalue problem
### Maintenance
- [ ] add: `.github/workflows/` directory for CI/CD
    - [ ] add: `format.yml` for formatting with `black` and `isort`
    - [ ] add: `lint.yml` for linting with `pylint` and `mypy`
    - [ ] add: `test.yml` for running tests with `pytest`
    - [ ] add: `doc.yml` for building documentation with `mkdocs`
- [ ] replace the default handling of traces in `LocalFunction` class with
  `DirichletTrace`
- [ ] deprecate `PiecewisePolynomial` class
- [ ] rename vertex and edge indices to something more descriptive (global mesh
  properties, not local to a cell)
- [ ] move `jacobi_preconditioner()` static method of `nystrom` class to
  separate module
- [ ] reduce redundant/duplicate code in computation of logarithmic terms,
  especially anti-Laplacians of logarithmic functions
- [ ] use binary exponentiation algorithm for `Polynomial.pow()`
- [ ] use `set` instead of `list` for vertex and cell indices in `PlanarMesh`
  class
- [ ] eliminate redundant calls to parameterize in `ClosedContour.parameterize()`
- [ ] `Monomial`:
  - [ ] replace `eval()` method with `__call__()` method
  - [ ] allow for `Floatlike` output from `eval()` method
  - [ ] deprecate `is_zero()` method
- [ ] `MultiIndex`:
  - [ ] use `Tuple` instead of `List` to set multi-index
- [ ] use sets instead of lists of `Monomial` objects in `Polynomial` class
- [ ] deprecate `barycentric_products()` function
- [ ] allow `barycentric_coordinates()` to accept a list of `Vert` objects
- [ ] weighted normal derivative of `Polynomial` objects in separate module
- [ ] fix disabled pylint messages
### Known bugs
- [ ] `Polynomial` not recognized as a callable map when passed to
  `is_Func_R2_R()`
- [ ] case with xy2 < TOL near corners on distinct edges in double layer
  operator can result in `nan` entries for large values of the discretization
  parameter `n`


## Tentative Features
These are features that are not yet planned, but are under consideration.
### Features
- [ ] automatic mesh generation and refinement
    - [ ] automatic determination of mesh cells from list of mesh edges (user
      does not need to specify `cell_idx` of each edge)
    - [ ] set vertex and edge indices automatically (maybe in `PlanarMesh`
      class?)
    - [ ] automatic identification of repeated cells in a mesh (up to scaling
      and rigid motion)
    - [ ] *a posteriori* error estimation
    - [ ] p-refinement: different polynomial degree on different cells/edges
- [ ] $H$(div)-conforming spaces
- [ ] $H$(curl)-conforming spaces
- [ ] surface elements
- [ ] 3D elements
### Examples
- [ ] time-dependent problems
- [ ] nonlinear problems
- [ ] Stokes flow
- [ ] Maxwell's equations
- [ ] shape optimization
### Maintenance
- [ ] type validation with `pydantic`
- [ ] add logging with `logging` module
- [ ] save and load `PlanarMesh` objects to/from file
- [ ] interior value interpolation for points close to the boundary
- [ ] modify `LocalFunctionSpace` to treat a split edge as a single edge
- [ ] add batch processing for multiple `LocalFunctions`
- [ ] use multiprocessing to speed up computation