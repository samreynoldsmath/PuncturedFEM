# Punctured FEM: To-do List


## [yyyy mm dd] v0.4
### Features
- [ ] global boundary conditions
  - [ ] zero Dirichlet
  - [ ] zero Neumann
  - [ ] mixed Dirichlet/Neumann
- [ ] file management
  - [ ] add saving and loading of `mesh` objects
  - [ ] add saving and loading local stiffness/mass matrices
### Optimizations
- [ ] D2N improvements
  - [ ] trigonometric interpolation
  - [ ] multiprocessing for batch computation
- [ ] `locfun.poly.poly`:
  - [ ] use binary exponentiation in `poly.pow()` method
- [ ] Mesh improvements
  - [ ] identify repeat edges
  - [ ] identify repeat cells
### Documentation
- [ ] README improvements
  - [ ] extend description
  - [ ] include references
- [ ] add installation guide
- [ ] replace `WISHLIST.md` with more comprehensive roadmap
- [ ] add contributing guide
### PyPI
- [ ] add `setup.py` file
- [ ] upload to PyPI
### Maintenance
- [ ] add logging with `logging` module
- [ ] use enumerate to replace `range(len(...))` loops
- [ ] use `functools.partial` to handle logarithmic functions
- [ ] rectify nested min/max statements
- [ ] move interior value calculation in `locfun` to separate module
- [ ] initialize solver in `locfunspace` and pass it to `antilap`
- [ ] `locfun.d2n` and `locfun.antilap`:
  - [ ] logarithmic functions moved to own module
  - [ ] rational functions moved to own module
- [ ] `locfun.poly.poly`:
  - [ ] use enum for `x` and `y` variable references (gradient etc.)
- [ ] `locfun.poly.piecewise_poly`
  - [ ] determine `num_polys` automatically in `piecewise_poly` constructor
- [ ] `mesh.cell`:
  - [ ] clean up integration methods
- [ ] `mesh.edge`:
  - [ ] safety checks for `set_cells` method
- [ ] `mesh.planar_mesh`:
  - [ ] replace lists with sets where appropriate
  - [ ] set vert idxs in constructor
- [ ] `mesh.edgelib.teardrop`:
  - [ ] pass `alpha` as keyword argument
- [ ] `plots`:
  - [ ] move quadrature dictionary to `mesh.quad` module
  - [ ] move plotting functions in `solver.solver` class to separate module
  - [ ] update examples to reflect change
- [ ] `solver.solver`:
  - [ ] move color printing to separate module

## [yyyy mm dd] v0.5
### Features
- [ ] Advection terms
- [ ] Diffusion terms
- [ ] Different material properties across regions
- [ ] Different polynomial degree on different cells/edges