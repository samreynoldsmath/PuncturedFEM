[2023 May 01] 0.2.5 -- API & polynomial overhaul
---
* Moved examples to a dedicated `examples` directory
  * Added "Examples used in publications" to `README`
* Introduced API for more convenient user experience
  * Users call `pf.thing` rather than `pf.foo.bar.baz.thing`
  * Updated all examples to reflect API change
  * Restructured `puncturedfem` directory
* Polynomial overhaul
  * Changed initialization method to something sensible
  * `set()` method behaves as a re-initializer
  * Added support for polynomial-scalar addition for `+`,`+=`,`-`,`-=` operators
  * Added support for `*=` operator
  * Moved `integrate_over_cell()` method to `locfun.int_poly`
* Unit tests
  * Relocated `test` to parent directory
  * Polynomials: `test_poly.py`
  * Local functions: `test_locfun.py`

[2023 Apr 17] 0.2.4 -- Interior values
---
* Added `intval` module to `locfun` for computing interior values
* Added `get_conjugable_part()` method to `locfun` that returns the trace of $\psi$
* Renamed `is_in_interior()` method for `contour` to `is_in_interior_contour()`
* Added `is_in_interior_cell()` method to `cell`
* Added interior value demo to Punctured Square, Pac-Man, and Ghost examples

[2023 Mar 06] 0.2.3 -- Pac-Man & Ghost
---
* Added **ex1b-pacman.ipynb**
* Added **ex1c-ghost.ipynb**
* Updated **ex1a-square-hole.ipynb** with more accurate reference values

[2023 Mar 01] 0.2.2 -- Anti-Laplacians
---
* Added `compute_h1()` and `compute_l2()` methods to `locfun`
* Added `integrate_over_cell()` method to `polynomial`
* Added methods to `contour` and `cell` to integrate over the boundary without multiplying by the norm of the derivative (i.e. 'preweighted' integrands)
* Monomials now default to zero monomial
* Fixed evaluation of zero `monomial` and zero `polynomial` to return same size as input
* Removed `ext_pt` field from `cell`
* Added `id` field to `contour`
* Added `get_integrator()` method to `contour` (`cell` inherits)
* Added `neumann` module to `nystrom` for solving Neumann problem
  * Modified `harmconj` module accordingly
* Completely overhauled anti-Laplacian calculation for punctured cells
* **ex1**: "Square with circular hole" updated

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
