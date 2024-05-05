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
## Documentation
- [ ] add: mathematical background
### Project Management
- [ ] add: `.github/workflows/` directory for CI/CD
  - [ ] add: `format.yml` for formatting with `black` and `isort`
  - [ ] add: `lint.yml` for linting with `pylint` and `mypy`
  - [ ] add: `test.yml` for running tests with `pytest`
  - [ ] add: `doc.yml` for building documentation with `mkdocs`
### Maintenance
- [ ] fix disabled pylint messages
- [ ] `LocalFunction` improvements:
  - [ ] interior value interpolation for points close to the boundary
  - [ ] replace the default handling of traces in `LocalFunction` class with `DirichletTrace` improvements:
  - [ ] reduce redundant/duplicate code in computation of logarithmic terms,
  especially anti-Laplacians of logarithmic functions
  - [ ] move weighted normal and tangential derivatives into `DirichletTrace`
  class, simplify the `LocalFunction` class
  - [ ] move `jacobi_preconditioner()` static method of `NystromSolver` class to
- [ ] `Polynomial` improvements:
  - [ ] deprecate `PiecewisePolynomial` class
  - [ ] replace `eval()` method with `__call__()` method
  - [ ] allow for `Floatlike` output from `eval()` method
  - [ ] use sets instead of lists of `Monomial` objects in `Polynomial` class
  - [ ] deprecate `barycentric_products()` function
  - [ ] allow `barycentric_coordinates()` to accept a list of `Vert` objects
  - [ ] use binary exponentiation algorithm for `Polynomial.pow()`
  - [ ] weighted normal derivative of `Polynomial` objects in separate module
- [ ] `Monomial` improvements:
  - [ ] replace `eval()` method with `__call__()` method
  - [ ] allow for `Floatlike` output from `eval()` method
  - [ ] deprecate `is_zero()` method
- [ ] `MultiIndex` improvements:
  - [ ] use `Tuple` instead of `List` to set multi-index
- [ ] `PlanarMesh` improvements:
  - [ ] rename vertex and edge indices to something more descriptive (global mesh
  properties, not local to a cell)
  - [ ] use `set` instead of `list` for vertex and cell indices in `PlanarMesh`
  class
  - [ ] eliminate redundant calls to parameterize in `ClosedContour.parameterize()`
  separate module

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
- [ ] modify `LocalFunctionSpace` to treat a split edge as a single edge
- [ ] add batch processing for multiple `LocalFunctions`
- [ ] use multiprocessing to speed up computation