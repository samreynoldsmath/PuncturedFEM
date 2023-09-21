# Punctured FEM: To-do List


## [yyyy mm dd] v0.3.x: Exceptions and logging
- rename `id` variables to avoid shadowing built-ins
- change exceptions to specific error types
- add logging with `logging` module


## [yyyy mmm dd] v0.3.x: Documentation
- write docstrings
  - classes
  - methods
  - functions
  - modules


## [yyyy mm dd] v0.3.x: Commit hooks
- `isort`
- `black`
- `mypy`
- `flake8`
- `pylint`
- `pytest`
- convert Jupyter notebooks to Python scripts


## [yyyy mm dd] v0.3.x: README improvements
- extend description
  - include references
- add installation guide
- add roadmap
- add contributing guide
- move affiliations to contributors section


## [yyyy mm dd] v0.3.x: D2N improvements
  - trigonometric interpolation
  - multiprocessing for batch computation


## [yyyy mm dd] v0.3.x: Refactor and clean up
- replace lists with sets where appropriate
- logarithmic and rational functions
- plots


## [yyyy mm dd] v0.4.x: Mesh improvements
  - identify repeat edges
  - identify repeat cells


## [yyyy mm dd] v0.4.x: Advection terms
  - implement $\int_K (b \cdot \nabla v) \, w ~dx$


## [yyyy mm dd] v0.4.x: Diffusion terms:
  - implement $\int_K (A \nabla v) \cdot \nabla w ~dx$
  - special case for $H^1$ semi-inner product ($A = I$)
  - special cases for harmonic functions and polynomials