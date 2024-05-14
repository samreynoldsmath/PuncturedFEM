# Roadmap

## [v0.6.0] - coming soon

### Features
- [ ] `PlanarMesh` file I/O
  - [ ] save and load `PlanarMesh` objects to/from file
  - [ ] convert `svg` files to meshes
- [ ] new system for defining `Edge` objects
  - [ ] curve type: line
  - [ ] curve type: spline
  - [ ] curve type: quadratic Bézier
  - [ ] curve type: cubic Bézier
  - [ ] curve type: symbolic (using `sympy`)

### Documentation
- [ ] update examples and tutorials to reflect new edge system
- [ ] tutorials
    - [ ] Local function spaces
        - [ ] 2.1 Polynomials
        - [ ] 2.3 Local functions (simpler version of Example 1.1)
        - [ ] 2.4 Local function spaces
    - [ ] FEM solver
        - [ ] 3.1 Global function spaces (simpler version of Example 2.1)
        - [ ] 3.2 Global boundary conditions

### Maintenance
- [ ] `GlobalFunctionSpace` improvements:
  - [ ] compute `EdgeSpace` objects only on reference edges
  - [ ] compute quantities of interest only on reference cells
- [ ] `PlanarMesh` improvements:
  - [ ] deprecate mesh builder functions in favor of file I/O system
  - [ ] automatic identification of repeated edges
  - [ ] automatic identification of repeated cells in a mesh (up to scaling
    and rigid motion)
- [ ] refactor `Quad`:
  - [ ] should be named `Quadrature` with sensible names for attributes
  - [ ] use `Quadrature` class to store all three types of quadrature rules
  - [ ] deprecate `QuadDict` class and `get_quad_dict()` function
  - [ ] the casual user should not need to touch `Quadrature` objects directly

### Project Management
- [ ] add `sympy` to dependencies

## Other Planned Features and Improvements

### Features
- [ ] global boundary conditions
    - [ ] define trace of a global function with `DirichletTrace` class
    - [ ] mixed Dirichlet/Neumann boundary conditions
    - [ ] nonhomogeneous boundary conditions
- [ ] *a posteriori* error estimation
- [ ] `BilinearForm` improvements
    - [ ] piecewise constant coefficients, load function
    - [ ] advection terms
    - [ ] diffusion terms
- [ ] $p$-refinement: different polynomial degree on different cells/edges
- [ ] $h$-refinement: algorithm to split cells

### Examples
- [ ] "subdivision refinement"
- [ ] eigenvalue problem

### Documentation
- [ ] add: mathematical background

### Maintenance
- [ ] fix disabled pylint messages
- [ ] `Polynomial` improvements:
  - [ ] deprecate `PiecewisePolynomial` class
  - [ ] deprecate `eval()` method, use `__call__()` method instead
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
  - [ ] set vertex and edge indices automatically
  - [ ] rename vertex and edge indices to something more descriptive (global mesh
  properties, not local to a cell)
  - [ ] use `set` instead of `list` for vertex and cell indices in `PlanarMesh`
  class
  - [ ] eliminate redundant calls to parameterize in `ClosedContour.parameterize()`
  separate module

### Project Management
- [ ] add: `.github/workflows/` directory for CI/CD
  - [ ] add: `format.yml` for formatting with `black` and `isort`
  - [ ] add: `lint.yml` for linting with `pylint` and `mypy`
  - [ ] add: `test.yml` for running tests with `pytest`
  - [ ] add: `doc.yml` for building documentation with `mkdocs`
- [ ] `git` hooks
  - [ ] add: `pre-commit` hook for formatting
    - [ ] `isort`
    - [ ] `black`
  - [ ] add: `pre-push` hook for linting and testing
    - [ ] `pylint`
    - [ ] `mypy`
    - [ ] `pydocstyle`
    - [ ] `pytest`
- [ ] update `CONTRIBUTING.md` with how to use `git` hooks and CI/CD

### Known bugs
- [ ] `Polynomial` not recognized as a callable map when passed to
  `is_Func_R2_R()`
- [ ] case with `xy2 < TOL` near corners on distinct edges in double layer
  operator can result in `nan` entries for large values of the discretization
  parameter `n`


## Tentative Features

### Features
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
- [ ] automatic identification of cells (user does not need to specify cell indices when creating an `Edge` object)
- [ ] allow sampling parameter `n` to have different values for different edges
- [ ] type validation with `pydantic`
- [ ] add logging with `logging` module
- [ ] modify `LocalFunctionSpace` to treat a split edge as a single edge
- [ ] add batch processing for multiple `LocalFunctions`
- [ ] use multiprocessing to speed up computation