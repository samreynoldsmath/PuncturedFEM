# Punctured FEM: To-do List

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
- [ ] Stokes flow
- [ ] Maxwell's equations
- [ ] Shape optimization

---
## v0.2.0: Python overhaul

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
### Examples: Jupyter notebooks
- [x] **ex0**: Edge and cell tutorial
- [x] **ex1**: D2N map and anti-Laplacians
  - [x] Computation of $H^1$ semi-inner product
  - [x] Computation of $L^2$ inner product

---
## v0.2.1: Local functions

### Polynomials
  - [x] `multi_index` objects
  - [x] `monomial` objects
  - [x] `polynomial` objects
    - [x] printing
    - [x] evaluation
  - [x] arithmetic operations
    - [x] addition
    - [x] multiplication
  - [x] calculus operations
    - [x] gradient
    - [x] laplacian
    - [x] anti-laplacian
### Local function object (elements of $V_p(K)$)
  - [x] Dirichlet trace
  - [x] Laplacian polynomial
  - [x] harmonic part $\phi$
  - [x] conjugate $\widehat\psi$
  - [x] logarithmic coefficients
  - [x] weighted normal derivative
    - [x] harmonic part wnd
    - [x] polynomial part wnd
### Cells
  - [x] shift to first quadrant

---
## v0.2.2: Anti-Laplacians

### Anti-Laplacians
  - [x] determine rational function coefficients
  - [x] set and solve systems for $\rho,\widehat\rho$
### Examples:
  - [x] **ex1**: Square with circular hole (update)
  - [ ] **ex2**: Pac-Man
  - [ ] **ex3**: Ghost

---
## v0.2.3: Interior values

### Organization
  - [ ] Consolidate `locfun`
    - [ ] `antilap`
    - [ ] `d2n`
    - [ ] `nystrom`
    - [ ] `poly`
  - [ ] Cosolidate `quad` under `mesh` (maybe...)
### Unit tests
  - [ ] anti-Laplacians
  - [ ] polynomials
  - [ ] interior values
### Interior values
  - [ ] generate points for evaluation
  - [ ] reduced interior domain
  - [ ] find points inside a closed contour
  - [ ] Cauchy's integral formula
    - [ ] values
    - [ ] gradient
  - [ ] plots
  - [ ] write to file
  - [ ] load from file
### Cells
  - [ ] boundary plotting method wrapper
  - [ ] contour orientation check (using rotation index computed from curvature)
### Local functions
  - [ ] contour plotting method wrapper

---
## v0.2.4: Nystrom Solver Optimization

### Quadrature
  - [ ] Simpson's rule
### Nyström Solver
  - [ ] set up and solve both types of systems with consolidated overhead
    - [ ] abstraction of Dirchlet, Neumann, and Robin boundary value problems
    - [ ] `set_up_right_hand_side` abstract method
    - [ ] `set_up_linear_operator` abstract method
    - [ ] precompute double and single layer operators
  - [ ] trigonometric interpolation
  - [ ] multiprocessing for batch computation
  - [ ] debug options
    - [ ] `gmres` flag
    - [ ] condition number
    - [ ] singular values

---
## v0.2.5: Bilinear forms

### Bilinear form
  - [ ] $L^2$ inner products: $\int_K v \, w ~dx$
  - [ ] advection terms :$\int_K (b \cdot \nabla v) \, w ~dx$
  - [ ] diffusion terms:
    - [ ] $\int_K (A \nabla v) \cdot \nabla w ~dx$
    - [ ] special case for $H^1$ semi-inner product ($A = I$)
  - [ ] special cases for harmonic functions and polynomials
### Examples: Jupyter notebooks
  - [ ] **ex2**: bilinear form evaluation

---
## v0.2.6: Local Function Spaces
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

---
## v0.3: Meshes

### Mesh
  - [ ] mesh encoding
  - [ ] write to file
  - [ ] load from file
### Global function space
  - [ ] assembler
  - [ ] solver
  - [ ] plots
### Examples: Jupyter notebooks
  - [ ] Mesh tutorial
  - [ ] Pegboard mesh: cells with 1, 4, 16 holes
  - [ ] Modified pegboard: shuriken and bean punctures
  - [ ] Nested annuli mesh
