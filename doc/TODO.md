# Punctured FEM: To-do List


## [yyyy mm dd] v0.3.x: Refactor and clean up
- [ ] move interior value calculation in `locfun` to separate module
- [ ] initialize solver in `locfunspace` and pass it to `antilap`
- [ ] `locfun.d2n` and `locfun.antilap`:
  - [ ] logarithmic functions moved to own module
  - [ ] rational functions moved to own module
- [ ] `locfun.poly.poly`:
  - [ ] use binary exponentiation in `poly.pow()`
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


## [yyyy mm dd] v0.3.x: README improvements
- [ ] extend description
  - [ ] include references
- [ ] add installation guide
- [ ] add roadmap
- [ ] add contributing guide


## [yyyy mm dd] v0.3.x: D2N improvements
- [ ] trigonometric interpolation
- [ ] multiprocessing for batch computation


## [yyyy mm dd] v0.3.x: Global boundary conditions
- [ ] zero Dirichlet
- [ ] zero Neumann
- [ ] mixed Dirichlet/Neumann


## [yyyy mm dd] v0.3.x: File management
- [ ] add saving and loading of `mesh` objects
- [ ] add saving and loading local stiffness/mass matrices


## [yyyy mm dd] v0.4.x: Mesh improvements
- [ ] identify repeat edges
- [ ] identify repeat cells


## [yyyy mm dd] v0.4.x: Advection terms
- [ ] implement $\int_K (b \cdot \nabla v) \, w ~dx$


## [yyyy mm dd] v0.4.x: Diffusion terms:
- [ ] implement $\int_K (A \nabla v) \cdot \nabla w ~dx$
- [ ] special case for $H^1$ semi-inner product ($A = I$)
- [ ] special cases for harmonic functions and polynomials