# Punctured FEM: To-do List

## v0.3.0: Meshes
### Edge function space
  - [x] Legendre polynomials and tensor products
  - [x] high order edge spaces
    - NOTE: still needs work to improve stability for high order > 3
  - [x] redundancy elimination
### Vertices & edges
- [x] `vert` class handles mesh vertices
- [x] overhaul `edge` class
  - [x] add mesh topology info
  - [x] initialize with `vert` pair
  - [x] parameterize as needed
### Cells
- [x] overhaul `cell` class
  - [x] add mesh topology into
  - [x] replace `edge` list with `closed_contour` list
  - [x] orientation of inner and outer boundaries
### Nyström Solver Refactor
  - [x] set up and solve both types of systems with consolidated overhead
  - [x] precompute double and single layer operators
### Mesh
- [x] `planar_mesh` class
  - [x] basic attributes and methods
- [x] identify cells from edge list
  - [x] orient edges
- [x] local-to-global map
### Local function space
  - [x] `locfun` handles trace as list of `polynomial` objects
  - [x] bubble functions
  - [x] vertex functions (only for edges with distinct endpoints)
  - [x] edge functions (includes case for vertex-free edges)
### Examples
  - [x] `ex0`: mesh building tutorial
  - [x] `ex1a/b/c`: local function quadrature
  - [x] `ex2a`: Pac-Man mesh
  - [x] generate `.py` files from Jupyter notebooks
### Tests
  - [x] pass

---
## v0.x.x: Optimizations
### Nyström Solver
  - [ ] trigonometric interpolation
  - [ ] multiprocessing for batch computation
### Mesh
  - [ ] identify repeat edges
  - [ ] identify repeat cells
### Refactor and clean up
- [ ] logartihmic and rational functions
- [ ] plots
### Examples
  - [ ] still work
### Tests
  - [ ] pass

---
## v0.x.x: Global Solver
### Global function space
  - [ ] bilinear form
  - [ ] assembler
  - [ ] solver
  - [ ] plots
### Examples
  - [ ] TBD
### Tests
  - [ ] pass

---
## v0.x.x: Bilinear forms
### Advection terms
  - [ ] $\int_K (b \cdot \nabla v) \, w ~dx$
### Diffusion terms:
  - [ ] $\int_K (A \nabla v) \cdot \nabla w ~dx$
  - [ ] special case for $H^1$ semi-inner product ($A = I$)
  - [ ] special cases for harmonic functions and polynomials
### Examples
  - [ ] TBD
### Tests
  - [ ] pass

---
## Planned features

### Mesh refinement
  - [ ] edge subdivision
    - [ ] reparameterization of new edges
  - [ ] cell subdivision
    - [ ] create two new cells by introducing an edge between vertices

### Interior value interpolation
  - [ ] Determine interior points close to boundary
  - [ ] Use boundary data to interpolate