# Roadmap


## Planned Features and Improvements
These are features that are planned for the future, but are not yet implemented.
### Features
- [ ] advection terms
- [ ] diffusion terms
- [ ] define an edge parameterization symbolically with `sympy`
- [ ] define trace of a global function with `DirichletTrace` class
- [ ] mixed zero Dirichlet/Neumann boundary conditions
- [ ] boundary interpolation and nonhomogeneous boundary conditions
- [ ] a posteriori error estimation
- [ ] p-refinement: different polynomial degree on different cells/edges
- [ ] piecewise constant coefficients and load functions in `BilinearForm`
### Examples
- [ ] add example of "subdivision refinement"
### Documentation
- [ ] add: table of contents
- [ ] add: tutorials
    - [ ] mesh construction
    - [ ] local function spaces
    - [ ] FEM solver
- [ ] add: mathematical background
### Maintenance
- [ ] replace the default handling of traces in `LocalFunction` class with `DirichletTrace`
- [ ] deprecate `PiecewisePolynomial` class
- [ ] modify `LocalFunctionSpace` to treat a split edge as a single edge
- [ ] rename vertex and edge indices to something more descriptive (global mesh properties, not local to a cell)
- [ ] set vertex and edge indices automatically (maybe in `PlanarMesh` class?)


## Tentative Features
These are features that are not yet planned, but are under consideration.
### Features
- [ ] automatic identification of repeated cells in a mesh (up to scaling and rigid motion)
- [ ] automatic mesh generation and refinement
- [ ] $H$(div)-conforming spaces
- [ ] $H$(curl)-conforming spaces
- [ ] surface elements
- [ ] 3D elements
### Examples
- [ ] eigenvalue problems
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