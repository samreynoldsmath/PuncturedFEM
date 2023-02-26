[2023 Feb 25] 0.2.1 -- Local functions
---
* Added **/poly** subpackage
  * `monomial` and `polynomial` objects
* Added **locfun/** subpackage
  * `locfun` object holds all data for local function $v\in V_p(K)$
* Added `ext_pt` field to `cell` object, which is an exterior point such that centering the origin at `ext_pt` placing the cell strictly in the first quadrant


[2023 Feb 16] 0.2.0 -- Python overhaul
---
* MATLAB code rewritten in Python to increase accessibility
* Examples presented with Jupyter Notebook
    - **ex0-mesh-building**:
      defining edges, cells, and meshes
    - **ex1-inner-prod**:
      compute $H^1$ and $L^2$ (semi-)inner products on punctured cell
* New subpackages/modules
    - **quad**: trapezoid, Kress, and Martensen quadrature objects
    - **mesh**: mesh construction tools
      - **edge**: parameterization of an edge
      - **contour**: collection of edges forming a simple closed contour
      - **cell**: mesh cell (subclass of contour)
    - **plot**: functions for
      - plot edges, contours, cell boundaries
      - trace of a function along a collection of edges
    - **nystrom**: Nystr$\text{\"o}$m method for solving integral equations
      - includes single and double layer operators
      - block system support
    - **d2n**: Dirichlet-to-Neumann map for harmonic functions
      - computation of harmonic conjugate
      - FFT (anti-)derivatives
    - **antilap**: tools to compute anti-Laplacians of harmonic functions
* Added unit tests to **puncturedfem/test/**
  - **test_harmconj** for harmonic conjugates
  - **test_fft_deriv** for FFT differentiation

[2022 Aug 02] 0.1.0 -- Initial commit
---
Only a simple diffusion operator (the Laplacian) is currently supported.
Dirichlet and mixed Dirichlet-Neumann boundary conditions are available,
but are assumed to be homogeneous. Used to run a simple "pegboard" example.
