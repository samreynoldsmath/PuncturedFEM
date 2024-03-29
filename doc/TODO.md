# Punctured FEM: To-do List


## Planned Features and Improvements
These are features that are planned for the future, but are not yet implemented.
### PyPI Release
- [ ] add `setup.py` file
- [ ] hook to upload new release to PyPI
### Documentation
- [ ] add installation guide to README (using `pip`)
### Examples
- [ ] add example of "subdivision refinement"
### Features
- [ ] define an edge parameterization symbolically with `sympy`
- [ ] define trace of a global function with `DirichletTrace` class
- [ ] mixed zero Dirichlet/Neumann boundary conditions
- [ ] boundary interpolation and nonhomogeneous boundary conditions
- [ ] a posteriori error estimation
- [ ] p-refinement: different polynomial degree on different cells/edges
- [ ] piecewise constant coefficients and load functions in `BilinearForm`
### Maintenance
- [ ] replace the default handling of traces in `LocalFunction` class with `DirichletTrace`
- [ ] deprecate `PiecewisePolynomial` class
- [ ] modify `LocalFunctionSpace` to treat a split edge as a single edge
- [ ] rename vertex and edge indices to something more descriptive (global mesh properties, not local to a cell)
- [ ] set vertex and edge indices automatically (maybe in `PlanarMesh` class?)


## Tentative Features
These are features that are not yet planned, but are under consideration.
### Examples
- [ ] eigenvalue problems
- [ ] time-dependent problems
- [ ] nonlinear problems
### Features
- [ ] automatic identification of repeated cells in a mesh (up to scaling and rigid motion)
- [ ] automatic mesh generation and refinement
- [ ] advection terms
- [ ] diffusion terms
- [ ] $H$(div)-conforming spaces
- [ ] $H$(curl)-conforming spaces
### Maintenance
- [ ] type validation with `pydantic`
- [ ] add logging with `logging` module
- [ ] save and load `PlanarMesh` objects to/from file