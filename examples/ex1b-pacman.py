#!/usr/bin/env python
# coding: utf-8

# # Example 1-B: Pac-Man
# 
# This is a continuation of Example 1-A. We will compute $L^2$ inner products
# and $H^1$ inner products of implicitly-defined functions on a punctured
# Pac-Man domain.

# In[ ]:


import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import puncturedfem as pf
import numpy as np
import matplotlib.pyplot as plt

# define quadrature schemes
n = 64
q_trap = pf.Quad(qtype="trap", n=n)
q_kress = pf.Quad(qtype="kress", n=n)
quad_dict = {"kress": q_kress, "trap": q_trap}

# define vertices
verts = []
verts.append(pf.Vert(x=0.0, y=0.0))
verts.append(pf.Vert(x=(np.sqrt(3) / 2), y=0.5))
verts.append(pf.Vert(x=(np.sqrt(3) / 2), y=-0.5))
verts.append(pf.Vert(x=-0.1, y=0.5))

# define edges
edges = []
edges.append(pf.Edge(verts[0], verts[1], pos_cell_idx=0))
edges.append(
    pf.Edge(
        verts[1],
        verts[2],
        pos_cell_idx=0,
        curve_type="circular_arc_deg",
        theta0=300,
    )
)
edges.append(pf.Edge(verts[2], verts[0], pos_cell_idx=0))
edges.append(
    pf.Edge(
        verts[3],
        verts[3],
        neg_cell_idx=0,
        curve_type="circle",
        radius=0.25,
        quad_type="trap",
    )
)

# define mesh cell
K = pf.MeshCell(idx=0, edges=edges)

# parameterize edges
K.parameterize(quad_dict)

# plot boundary
pf.plot.MeshPlot(K.get_edges()).draw()

# set up Nyström solver
nyst = pf.NystromSolver(K, verbose=True)


# ## Function with a gradient singularity
# 
# Consider the function
# \begin{align*}
# 	v(x) = r^\alpha \, \sin(\alpha \theta)
# \end{align*}
# where $x \mapsto (r, \theta)$ is given in polar coordinates,
# and $\alpha = 1/2$ is a fixed parameter. 
# Note that for $\alpha < 1$, 
# the gradient $\nabla v$ has a singularity at the origin.
# However, $v$ is harmonic everywhere else. 

# In[ ]:


# get Cartesian coordinates of points on boundary
x1, x2 = K.get_boundary_points()

# convert to polar
r = np.sqrt(x1**2 + x2**2)
th = np.arctan2(x2, x1) % (2 * np.pi)

# Dirichlet trace of v
alpha = 1 / 2
v_trace = r**alpha * np.sin(alpha * th)

# Laplacian of v (harmonic function)
v_lap = pf.Polynomial()

# build local function
v = pf.LocalFunction(nyst=nyst, lap_poly=v_lap, has_poly_trace=False)
v.set_trace_values(v_trace)

# compute all quantities needed for integration
v.compute_all()


# Note that the normal derivative is unbounded near the origin.
# Let's take a look at the weighted normal derivative.

# In[ ]:


pf.plot.TracePlot(
    traces=v.harm_part_wnd,
    title="Weighted normal derivative",
    fmt="k.",
    K=K,
    quad_dict=quad_dict,
).draw()


# ### $H^1$ seminorm
# 
# Let's try computing the square $H^1$ seminorm of $v$, 
# \begin{align*}
# 	\int_K |\nabla v|^2~dx
# 	&\approx 1.20953682240855912
# 	\pm 2.3929 \times 10^{-18}
# \end{align*}
# with an approximate value obtained with *Mathematica*.

# In[ ]:


h1_norm_sq_computed = v.get_h1_semi_inner_prod(v)
print("Computed square H^1 seminorm = ", h1_norm_sq_computed)

h1_norm_sq_exact = 1.20953682240855912
h1_norm_sq_error = abs(h1_norm_sq_computed - h1_norm_sq_exact)
print("Error in square H^1 seminorm = %.4e" % (h1_norm_sq_error))


# ### $L^2$ norm
# 
# Let's also try computing the $L^2$ norm
# \begin{align*}
# 	\int_K v^2 ~dx
# 	&\approx 0.97793431492143971
# 	\pm 3.6199\times 10^{-19}
# 	~.
# \end{align*}

# In[ ]:


l2_norm_sq_computed = v.get_l2_inner_prod(v)
print("Computed square L^2 seminorm = ", l2_norm_sq_computed)

l2_norm_sq_exact = 0.977934314921439713
l2_norm_sq_error = abs(l2_norm_sq_computed - l2_norm_sq_exact)
print("Error in square L^2 seminorm = %.4e" % l2_norm_sq_error)


# ### Convergence
# 
# Using Kress parameter $p=7$
# 
# |	n	|	H1 error	|	L2 error	|
# |-------|---------------|---------------|
# |	4	|	7.2078e-02	|	2.1955e-02	|
# |	8	|	3.3022e-02	|	5.4798e-03	|
# |	16	|	1.2495e-03	|	1.0159e-04	|
# |	32	|	6.5683e-06	|	4.6050e-07	|
# |	64	|	4.6834e-08	|	2.1726e-09	|

# ## Bonus: Interior values
# 
# Not included in the paper.

# In[ ]:


y1 = K.int_x1
y2 = K.int_x2

v.compute_interior_values()

v_computed = v.int_vals
v_x1_computed = v.int_grad1
v_x2_computed = v.int_grad2

plt.figure()
plt.contourf(y1, y2, v_computed, levels=50)
plt.colorbar()
plt.title("Interior values of $v$")

plt.figure()
plt.contourf(y1, y2, v_x1_computed, levels=50)
plt.colorbar()
plt.title("First component of grad $v$")

plt.figure()
plt.contourf(y1, y2, v_x2_computed, levels=50)
plt.colorbar()
plt.title("Second component of grad $v$")

# convert to polar
r = np.sqrt(y1**2 + y2**2)
th = np.arctan2(y2, y1) % (2 * np.pi)

# exact values
cos_th = np.cos(th)
sin_th = np.sin(th)
cos_ath = np.cos(alpha * th)
sin_ath = np.sin(alpha * th)

v_exact = r**alpha * sin_ath
v_x1_exact = alpha * r ** (alpha - 1) * (cos_th * sin_ath - sin_th * cos_ath)
v_x2_exact = alpha * r ** (alpha - 1) * (sin_th * sin_ath + cos_th * cos_ath)

# interior value errors
v_error = np.log10(np.abs(v_computed - v_exact))
plt.figure()
plt.contourf(y1, y2, v_error, levels=50)
plt.colorbar()
plt.title("Interior errors ($\log_{10}$)")

# first component of gradient errors
v_x1_error = np.log10(np.abs(v_x1_computed - v_x1_exact))
plt.figure()
plt.contourf(y1, y2, v_x1_error, levels=50)
plt.colorbar()
plt.title("Gradient errors in $x_1$ ($\log_{10}$)")

# second component of gradient errors
v_x2_error = np.log10(np.abs(v_x2_computed - v_x2_exact))
plt.figure()
plt.contourf(y1, y2, v_x2_error, levels=50)
plt.colorbar()
plt.title("Gradient errors in $x_2$ ($\log_{10}$)")

plt.show()

