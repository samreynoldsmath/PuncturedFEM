# Punctured FEM: To-do List


## v0.3.x: Documentation and type hints
- [ ] write docstrings
  - [ ] classes
  - [ ] methods
  - [ ] functions
  - [ ] modules
- [ ] add type hints


## v0.3.x: Private methods
- [ ] distinguish between public and private methods


## v0.3.x: Exceptions and logging
- [ ] change exceptions to specific error types
- [ ] add logging


## v0.3.x: Commit hooks
- [ ] write commit checklist
- [ ] `isort`
- [ ] `mypy`
- [ ] `black`
- [ ] `flake8`
- [ ] `pylint`
- [ ] `pytest`
- [ ] convert Jupyter notebooks to Python scripts


## v0.4.x: Optimizations
### Sets
- [ ] replace lists with sets where appropriate
### Nyström Solver
  - [ ] trigonometric interpolation
  - [ ] multiprocessing for batch computation
### Mesh
  - [ ] identify repeat edges
  - [ ] identify repeat cells
### Refactor and clean up
- [ ] logarithmic and rational functions
- [ ] plots

## v0.x.x: Bilinear forms
### Advection terms
  - [ ] $\int_K (b \cdot \nabla v) \, w ~dx$
### Diffusion terms:
  - [ ] $\int_K (A \nabla v) \cdot \nabla w ~dx$
  - [ ] special case for $H^1$ semi-inner product ($A = I$)
  - [ ] special cases for harmonic functions and polynomials

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