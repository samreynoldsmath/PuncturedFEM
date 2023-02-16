# Punctured FEM: To-do List

---
## v0.2: Python overhaul

### Quadrature
  - [x] trapezoid
  - [x] Kress
  - [x] Martensen
  - [x] $2n+1$ sampled points (to allow for edge orientation flipping)
### Edges
  - [x] points on the boundary
  - [x] norm of derivative (don't forget chain rule)
  - [x] unit tangential and normal vectors
  - [x] curvature (signed)
  - [x] `duplicate` method: create second instance of edge object
  - [x] `dialate` method
  - [x] `rotate` method
  - [x] `translate` method
  - [x] `flip_orientation` method
  - [x] `set_endpoints` method: set the endpoints
  - [x] `apply_orthogonal_transformation` method
  - [x] method to check equality between two edges
### Edge library
  - [x] circle (closed contour)
  - [x] bean shape (closed contour)
  - [x] teardrop (closed contour)
  - [x] line
  - [x] sine wave
  - [x] circular arc
### Mesh cell
  - [x] each `cell` object is an ordered list of edges
  - [x] method to evaluate a function over entire boundary
  - [x] automatically identify closed contours
  - [x] method to find index of nearest vertex
    - [x] correct for edges listed in arbitrary order
  - [x] automatically choose point $\xi_j$ in the interior of $j$th hole
  - [x] automatically choose a point $\xi_0$ outside of the "base domain" $K_0$
### Basic Plots
  - [x] plot boundary
  - [x] plot oriented boundary
  - [x] plot trace
  - [x] plot trace with logarithmic scale
### Nyström Solver
  - [x] single layer operator
    - [x] denest loops over edges
    - [x] compatible with trapezoid and Kress quadratures
  - [x] double layer potential operator
    - [x] denest loops over edges
    - [x] compatible with trapezoid and Kress quadratures
### Harmonic conjugates: set up and solve Neumann problem
  - [x] simply-connected
  - [x] logarithmic traces, normal and tangential deriviatives
  - [x] multiply-connected
  - [x] unit tests
### Dirichlet-to-Neumann map
  - [x] FFT derivative
  - [x] Dirichlet-to-tangential map
  - [x] Dirichlet-to-Neumann map
  - [x] unit tests
### Anti-Laplacians
  - [x] simply-connected cells: FFT anti-derivative
  - [x] multiply-connected cells: solve Robin problem
    - [x] center at point outside domain
  - [ ] unit tests
### Examples: Jupyter notebooks
- [x] **ex0**: Edge and cell tutorial
- [x] **ex1**: D2N map and anti-Laplacians
  - [x] Computation of $H^1$ semi-inner product
  - [x] Computation of $L^2$ inner product

---
## v0.3: Bilinear form update

### Polynomials
  - [ ] `poly` class definition
    - [ ] initialize to zero polynomial
    - [ ] index management
    - [ ] origin recentering
    - [ ] printing
    - [ ] evaluation
  - [ ] arithmetic operations
    - [ ] addition
    - [ ] multiplication
  - [ ] calculus operations
    - [ ] gradient
    - [ ] laplacian
    - [ ] anti-laplacian
  - [ ] `polyvec` class: polynomial vectors
    - [ ] divergence
### Trace object
  - [ ] abstraction of Dirichlet trace, normal and tangential derivatives
### Local function object (elements of $V_p(K)$)
  - [ ] Dirichlet trace
  - [ ] Laplacian polynomial
  - [ ] harmonic part
  - [ ] conjugable part
  - [ ] conjugate
  - [ ] logarithmic coefficients
  - [ ] tangential derivative
  - [ ] normal derivative
  - [ ] method to obtain interior value (w/ Cauchy's integral formula)
### Interior values
  - [ ] generate points for evaluation
  - [ ] reduced interior domain
  - [ ] find points inside a closed contour
  - [ ] write to file
  - [ ] load from file
  - [ ] plots
### Bilinear form
  - [ ] $L^2$ inner products: $\int_K v \, w ~dx$
  - [ ] advection terms :$\int_K (b \cdot \nabla v) \, w ~dx$
  - [ ] diffusion terms:
    - [ ] $\int_K (A \nabla v) \cdot \nabla w ~dx$
    - [ ] special case for $H^1$ semi-inner product ($A = I$)
  - [ ] special cases for harmonic functions and polynomials
### Examples: Jupyter notebooks
  - [ ] **ex1**: rework with `local_function` object
    - [ ] Interior values

---
## v0.4: Local function space update

### Quadrature
  - [ ] Simpson's rule
### Edge function space
  - [ ] special case for closed contours
  - [ ] barycentric coordinates (?)
  - [ ] arbitrary polynomial degree (integrated Legendre polynomials)
  - [ ] redundancy elimination
### Local function space
  - [ ] vertex functions
  - [ ] edge functions
  - [ ] bubble functions
  - [ ] local stiffness and mass matrices
### Nyström Solver
  - [ ] set up and solve both types of systems with consolidated overhead
    - [ ] abstraction of Dirchlet, Neumann, and Robin boundary value problems
    - [ ] `set_up_right_hand_side` abstract method
    - [ ] `set_up_linear_operator` abstract method
  - [ ] trigonometric interpolation
  - [ ] multiprocessing for batch computation
  - [ ] debug options
    - [ ] `gmres` flag
    - [ ] condition number
    - [ ] singular values
### Examples: Jupyter notebooks
  - [ ] Local function spaces

---
## v0.5: Mesh update

### Mesh
  - [ ] mesh encoding
  - [ ] write to file
  - [ ] load from file
### Global function space
  - [ ] assembler
  - [ ] solver
  - [ ] plots
### Examples: Jupyter notebooks
  - [ ] Local function spaces
  - [ ] Mesh tutorial
  - [ ] Pegboard mesh: cells with 1, 4, 16 holes
  - [ ] Modified pegboard: shuriken and bean punctures
  - [ ] Nested annuli mesh

---
## Features Wishlist
- [ ] Nonhomogeneous Dirichlet/Neumann boundary conditions
- [ ] Different material properties across regions
- [ ] Different polynomial degree on different cells/edges
- [ ] $H$(div) and $H$(curl) spaces
- [ ] Automatic mesh generation/refinement
- [ ] Surface elements
- [ ] 3D

---
## Applications Wishlist
- [ ] Eigenvalue problems
- [ ] Time-dependent problems
- [ ] Inverse scattering
- [ ] Stokes flow
- [ ] Maxwell's equations
- [ ] Shape optimization
