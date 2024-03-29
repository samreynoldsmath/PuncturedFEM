#!/usr/bin/env python
# coding: utf-8

# # Example 2: Finite Elements on a Pac-Man Mesh
# ### Sam Reynolds, 2023
# 
# This example demonstrates how to set up and solve a finite element problem on a
# punctured mesh. 
# The model problem under consideration is a simple diffusion-reaction problem
# \begin{align*}
# 	-\nabla\cdot(a \, \nabla u) + c \, u &= f \quad \text{in } \Omega, \\
# 	u &= 0 \quad \text{on } \partial\Omega,
# \end{align*}
# where $a, c$ are constant scalars and $f$ is a polynomial.
# The associated weak form is
# \begin{align*}
# 	\int_\Omega a \, \nabla u \cdot \nabla v \, dx
# 	+ \int_\Omega c \, u \, v \, dx
# 	&= \int_\Omega f \, v \, dx
# 	\quad \forall v \in H^1_0(\Omega).
# \end{align*}
# In previous examples, we saw that we can evalate these integrals on each cell
# $K$ in a mesh $\mathcal{T}$ of the domain $\Omega$, provided that $u$ and $v$
# are elements of a *local Poisson space* $V_p(K)$.
# We define the *global Poisson space* $V_p(\mathcal{T})$ as the space of
# continuous functions in $H^1_0(\Omega)$ whose restriction to each cell $K$ is
# an element of $V_p(K)$.
# By constructing a basis $\{\phi_1, \dots, \phi_N\}$ of $V_p(\mathcal{T})$ by 
# continuously "stitching" the local basis functions together,
# we seek a finite element solution $\tilde{u} \in V_p(\mathcal{T})$ such that
# \begin{align*}
# 	&\tilde{u} = \sum_{i=1}^N u_i \, \phi_i,
# 	\\
# 	&\int_\Omega a \, u_i \nabla \phi_i \cdot \nabla \phi_j \, dx
# 	+ \int_\Omega c \, u_i \, \phi_i \, \phi_j \, dx
# 	= \int_\Omega f \, \phi_j \, dx
# \end{align*}

# In[ ]:


import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import puncturedfem as pf


# Let's set a few parameters before we go any further. 
# `deg` is the polynomial degree of global Poisson space,
# `n` is edge sampling parameter (as used in previous examples).
# 
# **(!) WARNING:** 
# Higher order spaces (`deg > 1`) are still under development.

# In[ ]:


deg = 1
n = 64


# ## Mesh construction
# The mesh we will use for this example was constructed in 
# [Example 0](ex0-mesh-building.ipynb).
# For convenience, the same mesh can be constructed by calling the `pacman_mesh`
# function in the `mesh.meshlib` module.

# In[ ]:


T = pf.meshlib.pacman_subdiv()


# Let's take a look at the mesh by using the `MeshPlot` class.

# In[ ]:


pf.plot.MeshPlot(T.edges, n).draw(show_axis=False, pad=0.0)


# ## Build global function space 
# The global function space $V_p(\mathcal{T})\subset H^1(\Omega)$ 
# is the space of continuous functions such that each function belongs to 
# $V_p(K)$ when restricted to any cell $K\in\mathcal{T}$.
# (Note that we use `deg` to denote the polynomial degree $p$.)
# 
# To proceed with the computation, we define the quadrature scheme(s) used to 
# parameterize the edges of the mesh.

# In[ ]:


quad_dict = pf.get_quad_dict(n)


# The global function space `V` is built from the mesh `T`, along with the `deg`
# parameter and the information necessary to parameterize the edges of the mesh.

# In[ ]:


V = pf.GlobalFunctionSpace(T, deg, quad_dict)


# ## Define a bilinear form
# The bilinear form 
# \begin{align*}
# 	B(u,v) = 
# 	\int_\Omega a \, \nabla u \cdot \nabla v ~dx
# 	+ \int_\Omega c \, u \, v ~dx
# \end{align*}
# and the right-hand side linear functional
# \begin{align*}
# 	F(v) = \int_\Omega f \, v ~dx
# \end{align*}
# are declared as follows,
# with `diffusion_coefficient` $a = 1$, 
# `reaction_coefficient` $c = 1$,
# and `rhs_poly` $f(x) = 1 \cdot x^{(0, 0)}$.

# In[ ]:


a = 1.0
c = 1.0
f = pf.Polynomial([(1.0, 0, 0)])

B = pf.BilinearForm(
    diffusion_constant=a,
    reaction_constant=c,
    rhs_poly=f,
)

print(B)


# ## Set up the finite element solver
# A finite element solver needs two things: the global function space and the bilinear form. 

# In[ ]:


solver = pf.Solver(V, B)


# To assemble the matrix and right-hand side vector for the global system, we 
# call the `assemble()` method.
# Zero Dirichlet boundary conditions are incorporated by default.
# 
# This can take a while. You may want to grab a cup of coffee.

# In[ ]:


solver.assemble()


# The `matplotlib.pyplot` module has a handy function for inspecting the sparsity
# pattern of a matrix.  Let's take a look at the global matrix.

# In[ ]:


import matplotlib.pyplot as plt

plt.figure()
plt.spy(solver.glob_mat)
plt.grid(True)
plt.show()


# ## Solving the global linear system
# To solve the system we worked hard to set up, we can call the `solve()` method
# on the `Solver` object.

# In[ ]:


solver.solve()


# ## Plot solution
# We can visualize the solution by 
# creating an instance of the `GlobalFunctionPlot` class.
# There are two types of plots available: 
# a conventional contour plot (`fill=False`)
# or a heat map (`fill=True`).
# To view the figure in this notebook, set `show_fig = True`.
# To save it to a file, set the `filename` keyword argument in the 
# `draw()` method.

# In[ ]:


cm = plt.colormaps["seismic"]
pf.plot.GlobalFunctionPlot(solver).draw(pad=0.0, show_axis=False, colormap=cm)


# ## Plot global basis functions
# Let's take a look at the global basis functions.

# In[ ]:


import numpy as np

for idx in range(V.num_funs):
    coef = np.zeros(V.num_funs)
    coef[idx] = 1.0
    pf.plot.GlobalFunctionPlot(solver, coef).draw(pad=0.0, show_axis=False)

