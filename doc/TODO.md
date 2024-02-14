# Punctured FEM: To-do List


## [yyyy mmm dd] v0.5
### Documentation
- [ ] add installation guide
- [ ] add contributing guide
### PyPI Release
- [ ] add `setup.py` file
- [ ] hook to upload new release to PyPI
### Features
- [ ] automatically identify repeated cells in a mesh (up to scaling and rigid motion)
- [ ] add ability to define an edge parameterization symbolically with `sympy`
- [ ] add ability to define an edge parameterization using a cubic spline to interpolate points
### Maintenance
- [ ] add logging with `logging` module


## Planned Features

### Features
- [ ] p-refinement: different polynomial degree on different cells/edges
- [ ] different material properties across regions
- [ ] add saving and loading of `PlanarMesh` objects to/from file
- [ ] add mixed zero Dirichlet/Neumann boundary conditions
- [ ] edge space interpolation
- [ ] nonhomogeneous boundary conditions
- [ ] a posteriori error estimation
- [ ] advection terms
- [ ] diffusion terms
### Examples
- [ ] add example of "subdivision refinement"
