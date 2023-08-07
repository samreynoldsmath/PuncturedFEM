#!/usr/bin/env python
# coding: utf-8

# # Example 1-A: Square with a Circular Hole
# 
# Given an open, bounded, connected region $K \subset \mathbb{R}^2$
# with a sufficiently "nice" boundary, let $v,w : K \to \mathbb{R}$
# be functions of the form
# \begin{align*}
# 	v = \phi + P~,
# 	\quad 
# 	w = \psi + Q
# \end{align*}
# where $\phi,\psi$ are harmonic functions and $P,Q$ are polynomials.
# The goal of this example is to compute the 
# $H^1$ semi-inner product and $L^2$ inner product
# \begin{align*}
# 	\int_K \nabla v \cdot \nabla w ~dx 
# 	~, \quad 
# 	\int_K v \, w ~dx
# \end{align*}
# using only 
# (i) the Dirichlet traces $v|_{\partial K}, w|_{\partial K}$, and
# (ii) the Laplacians $\Delta v, \Delta w$, which are polynomials.
# 
# Our strategy is to reduce these volumetric integrals over $K$
# to boundary integrals on $\partial K$.
# The procedure uses two key elements:
# given a harmonic function $\phi$, compute
# 1. the normal derivative $\frac{\partial\phi}{\partial\mathbf{n}} = \nabla\phi\cdot\mathbf{n}$
# 2. an anti-Laplacian $\Phi$ satisfying $\Delta\Phi = \phi$ 

# ## Define a Mesh Cell
# 
# We will take $K$ to be a unit square with a circular hole,
# and create a `cell` object accordingly. 
# See `ex0-mesh-building` for details.

# In[1]:


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
q_trap = pf.quad(qtype='trap', n=n)
q_kress = pf.quad(qtype='kress', n=n)
quad_dict = {'kress': q_kress, 'trap': q_trap}

# define vertices
verts: list[pf.vert] = []
verts.append(pf.vert(x=0.0, y=0.0))
verts.append(pf.vert(x=1.0, y=0.0))
verts.append(pf.vert(x=1.0, y=1.0))
verts.append(pf.vert(x=0.0, y=1.0))
verts.append(pf.vert(x=0.5, y=0.5))	# center of circle

# define edges
edges: list[pf.edge] = []
edges.append(pf.edge(verts[0], verts[1], pos_cell_idx=0))
edges.append(pf.edge(verts[1], verts[2], pos_cell_idx=0))
edges.append(pf.edge(verts[2], verts[3], pos_cell_idx=0))
edges.append(pf.edge(verts[3], verts[0], pos_cell_idx=0))
edges.append(pf.edge(verts[4], verts[4], neg_cell_idx=0,
         curve_type='circle', quad_type='trap', radius=0.25))

# define mesh cell
K = pf.cell(id=0, edges=edges)

# parameterize edges
K.parameterize(quad_dict)

# plot boundary
pf.plot_edges(edges, orientation=True)


# ## Define a Local Function
# 
# A local function $v \in V_p(K)$ can be uniquely defined by specifying its 
# Dirichlet trace $v|_{\partial K}$ and Laplacian $\Delta v$.
# In practice, $v$ would be implicitly defined in this way, but for the purpose 
# of testing our numerics, we will consider explicitly defined functions for this
# example.
# 
# ### Define a Dirichlet Trace
# 
# Consider the function $v$ given by 
# \begin{align*}
# 	v(x) = e^{x_1} \, \cos x_2 + a \ln|x-\xi| + x_1^3 x_2 + x_1 x_2^3
# 	~,
# \end{align*}
# where $a = 1$ and $\xi = (0.5, 0.5)$ is a point located in the hole of $K$.
# We see immediately that $v$ can be decomposed into a harmonic part $\phi$ and
# a polynomial part $P$:
# \begin{align*}
# 	v &= \phi + P~,
# 	\\ 
# 	\phi(x) &= e^{x_1} \, \cos x_2 + a \ln|x-\xi|~,
# 	\\
# 	P(x) &= x_1^3 x_2 + x_1 x_2^3
# 	~. 
# \end{align*}
# First, let's compute the values of the Dirichlet trace of $v$:

# In[2]:


# set target value of logarithmic coefficient
a_exact = 1.0

# set point in hole interior
xi = [0.5, 0.5]

# get the coordinates of sampled boundary points
x1, x2 = K.get_boundary_points()

# define trace of v
v_trace = np.exp(x1) * np.cos(x2) + \
    0.5 * a_exact * np.log((x1 - xi[0]) ** 2 + (x2 - xi[1]) ** 2) + \
    x1 ** 3 * x2 + x1 * x2 ** 3


# ### Define a Polynomial Laplacian
# 
# The Laplacian of $v$ is given by 
# \begin{align*}
# 	\Delta v(x) = \Delta P(x) = 12 x_1 x_2~.
# \end{align*}
# We will encode this an a `polynomial` object with a multi-index 
# $\alpha = (1,1)$ and coefficient $c_\alpha = 12$.

# In[3]:


# create polynomial object
v_laplacian = pf.polynomial([ [12.0, 1, 1] ])


# ### Define a Local Function
# 
# We are now ready to define $v$ as a `locfun` object:

# In[4]:


solver = pf.nystrom_solver(K)


# In[5]:


v = pf.locfun(solver, lap_poly=v_laplacian, has_poly_trace=False)
v.set_trace_values(v_trace)


# To proceed with our calculations, we must determine a polynomial anti-Laplacian
# of $\Delta v$, which we might expect to be $P(x) = x_1^3 x_2 + x_1 x_2^3$.

# In[6]:


v.compute_polynomial_part()
print(v.get_polynomial_part())


# **Remark.** The decomposition $v = \phi + P$ is not unique, since the 
# intersection between harmonic functions and polynomials is contains more than
# just the zero function (a lot more).
# The polynomial $P(x) = x_1^3 x_2 + x_1 x_2^3$ was chosen carefully for this example
# so that the computed anti-Laplacian of $\Delta P(x) = 12 x_1 x_2$ coincides 
# with $P$.
# This will not generally be the case.

# ## Find the Normal Derivative
# 
# Our first task is to compute the normal derivative of $v = \phi + P$.
# Recall that a *harmonic conjugate* of a harmonic function 
# $\psi$ is another harmonic function,
# which we will denote by $\widehat\psi$, for which the Cauchy-Riemann equations
# are satisfied:
# \begin{align*}
# 	\frac{\partial\phi}{\partial x_1} 
# 	= \frac{\partial\widehat\phi}{\partial x_2}
# 	~,\quad
# 	\frac{\partial\phi}{\partial x_2} 
# 	= -\frac{\partial\widehat\phi}{\partial x_1}
# 	~.
# \end{align*}
# It follows that the normal derivative of $\psi$ and the tangential derivative
# of $\widehat\psi$ are equal:
# \begin{align*}
# 	\frac{\partial\psi}{\partial\mathbf{n}} 
# 	= \frac{\partial\widehat\psi}{\partial\mathbf{t}}
# 	~.
# \end{align*}
# There is a minor issue when dealing with domains with holes:
# $\widehat\psi$ may not exist. Fortunately, we have a workaround.
# 
# **Logarithmic Conjugation Theorem.**
# Given a harmonic function $\phi$ and points $\xi_j$ 
# located in the interior of the $j$-th hole ($1\leq j\leq m$), 
# then there is a harmonic function $\psi$ with a harmonic 
# conjugate and real coefficients $a_1,\dots,a_m$ such that 
# \begin{align*}
# 	\phi(x) = \psi(x) + \sum_{j=1}^m a_m \ln|x-\xi_j|
# 	~.
# \end{align*}
# We will determine the trace of $\widehat\psi$ and the logarithmic 
# coefficients $a_1, \dots, a_m$ by solving an integral equation numerically.
# The user does not need to know the specifics, but merely needs to call
# the following two methods.

# In[7]:


v.compute_polynomial_part_trace()
v.compute_harmonic_conjugate()


# ### Error in Logarithmic Coefficient
# 
# Recall that 
# \begin{align*}
# 	\phi(x) = \psi(x) + a \ln|x-\xi|~,
# 	\quad 
# 	\psi(x) = e^{x_1} \, \cos x_2~,
# \end{align*}
# with $\widehat\psi(x) = e^{x_1} \, \sin(x_2)$ being a harmonic conjugate of 
# $\psi$.
# We chose the point $\xi = (0.5, 0.5)$ carefully for this problem,
# since this is also the interior point $\xi_1$ that was chosen automatically
# when we created $K$.
# Therefore, we ought to find that $a_1 = a = 1$, which we can check now:

# In[8]:


print('Computed logarithmic coefficient = ', v.log_coef[0])
print('Error = ', abs(v.log_coef[0] - a_exact))


# ### Error in Harmonic Conjugate Trace  
# 
# Recall that 
# $$
# 	\widehat\psi(x) = e^{x_1} \, \sin x_2
# $$
# is a harmonic conjugate of $\psi(x) = e^{x_1} \, \cos x_2$.
# Let's compare this to the computed trace of $\widehat\psi$.
# We can use call `plot.traceplot.trace()` to plot the trace(s) 
# of function(s) on the boundary.

# In[9]:


# get computed value of psi_hat
psi_hat_computed = v.get_harmonic_conjugate()

# get exact trace of psi_hat
psi_hat_exact = np.exp(x1) * np.sin(x2)

# plot harmonic conjugate
quad_list = [q_trap, q_kress,]
f_trace_list = [psi_hat_exact, psi_hat_computed,]
fmt = ('g--', 'k.')
legend = ('exact','computed')
title = 'Harmonic conjugate $\hat\psi$ of conjugable part of $\phi$'
pf.plot_trace(f_trace_list, fmt, legend, title, K, quad_list)


# **Note**: A harmonic conjugate is unique only up to an additive constant.
# So to compute the error in $\hat\psi$,
# we compute $\hat\psi_\text{exact} - (\widehat\psi_\text{computed} - c)$, 
# where $c$ is a constant that minimizes the $L^2(\partial K)$ norm,
# which is
# \begin{align*}
# 	c = -\frac{1}{|\partial K|}\int_{\partial K} 
# 	(\hat\psi_\text{exact} - \widehat\psi_\text{computed}) ~ds
# 	~.
# \end{align*}

# In[10]:


# average square distance between values
boundary_length = K.integrate_over_boundary(np.ones((K.num_pts,)))
integrated_difference = \
    K.integrate_over_boundary(psi_hat_exact - psi_hat_computed)
c = - integrated_difference / boundary_length

# plot harmonic conjugate
quad_list = [q_trap, q_kress,]
f_trace_list = [psi_hat_exact, psi_hat_computed - c ,]
fmt = ('g--', 'k.')
legend = ('exact','computed')
title = 'Harmonic conjugate $\hat\psi$ of conjugable part of $\phi$'
pf.plot_trace(f_trace_list, fmt, legend, title, K, quad_list)


# Compute and plot the error in the computed harmonic conjugate.

# In[11]:


# compute errors in harmonic conjugate
psi_hat_error = np.abs(psi_hat_exact - psi_hat_computed + c)

# plot harmonic conjugate error
f_trace_list = [psi_hat_error,]
fmt = ('k.',)
legend = ('error',)
title = 'Harmonic conjugate error'
pf.plot_trace_log(f_trace_list, fmt, legend, title, K, quad_list)


# The pointwise errors look alright. 
# Let's compute the $L^2(\partial K)$ norm of the error:

# In[12]:


max_hc_error = max(psi_hat_error)
l2_hc_error = np.sqrt(K.integrate_over_boundary(psi_hat_error ** 2))
print('Max pointwise error = %.4e'%max_hc_error)
print('L^2 norm of error = %.4e'%l2_hc_error)


# ### Compute the Normal Derivative
# 
# Recall that the Cauchy-Riemann equations imply that we can obtain the normal 
# derivative of $\psi$ using the tangential derivative of its harmonic conjugate:
# $$
# 	\dfrac{\partial\psi}{\partial\mathbf{n}} 
# 	= \dfrac{\partial\hat\psi}{\partial\mathbf{t}}
# 	~.
# $$
# Furthermore, if $x(t)$ is a parameterization of $\partial K$, we have 
# $$
# 	\dfrac{d}{dt}\hat\psi(x(t)) 
# 	= \dfrac{\partial\hat\psi(x(t))}{\partial\mathbf{t}} \, |x'(t)|
# 	~.
# $$
# We refer to this derivative as a **weighted tangential derivative**.
# Similarly, we refer to 
# \begin{align*}
# 	\dfrac{\partial\psi(x(t))}{\partial\mathbf{n}} \, |x'(t)|
# \end{align*}
# the a **weighted tangential derivative** of $\psi$.
# 
# **Remark.** Fortunately for us, $|x'(t)|$ appears as the Jacobian in the 
# integral
# \begin{align*}
# 	\int_{\partial K} \eta \, \frac{\partial\psi}{\partial\mathbf{n}}~ds
# 	=
# 	\int_a^b \eta(x(t)) \, 
# 	\dfrac{\partial\psi(x(t))}{\partial\mathbf{n}} \, |x'(t)| ~dt
# 	~,
# \end{align*}
# so we will be satisfied with the weighted normal derivative.
# 
# **Remark.** 
# If $\partial K$ is parameterized with a regular curve 
# (i.e. $|x'(t)| > 0$ for all $t$),
# then we can recover the normal derivative values. 
# However, this is not recommended when $K$ has corners, 
# as the reparameterization using Kress sampling is not regular, 
# leading to division-by-zero headaches.
# This is reflected in the fact that normal derivatives of harmonic functions
# are discontinuous when $\partial K$ has corners, and indeed the normal 
# derivative may even be unbounded.
# 
# Let's obtain the weighted normal derivative of $\phi$ by calling the 
# `compute_harmonic_weighted_normal_derivative()` method.

# In[13]:


# compute weighted normal derivative
v.compute_harmonic_weighted_normal_derivative()


# Note that the exact values of the normal derivative are given by
# \begin{align*}
# 	\frac{\partial\phi}{\partial\mathbf{n}} =
# 	\nabla\phi(x) \cdot \mathbf{n} = e^{x_1}
# 	\begin{pmatrix}
# 		\cos x_2 \\ -\sin x_2
# 	\end{pmatrix}
# 	\cdot \mathbf{n}
# 	+
# 	a \, \frac{(x-\xi) \cdot \mathbf{n}}{|x - \xi|^2}
# \end{align*}
# Let's compute these exact values for comparison.

# In[14]:


# define the components of the gradient of phi
phi_x1 = np.exp(x1) * np.cos(x2) + \
    a_exact * (x1 - xi[0]) / ((x1 - xi[0]) ** 2 + (x2 - xi[1]) ** 2)
phi_x2 = -np.exp(x1) * np.sin(x2) + \
    a_exact * (x2 - xi[1]) / ((x1 - xi[0]) ** 2 + (x2 - xi[1]) ** 2)

# compute exact weighted normal derivative
phi_nd = K.dot_with_normal(phi_x1, phi_x2)
phi_wnd_exact = K.multiply_by_dx_norm(phi_nd)

# get computed values
phi_wnd_computed = v.get_harmonic_weighted_normal_derivative()

# compute errors
wnd_error = np.abs(phi_wnd_computed - phi_wnd_exact)

# plot exact and computed weighted normal derivatives
quad_list = [q_trap, q_kress,]
f_trace_list = [phi_wnd_exact, phi_wnd_computed,]
fmt = ('g--', 'k.')
legend = ('exact','computed')
title = 'Weighted normal derivative'
pf.plot_trace(f_trace_list, fmt, legend, title, K, quad_list)

# plot errors
f_trace_list = [wnd_error,]
fmt = ('k.',)
legend = ('error',)
title = 'Weighted normal derivative error'
pf.plot_trace_log(f_trace_list, fmt, legend, title, K, quad_list)


# Let's look at the maximum pointwise error as well as the error in the
# $L^2(\partial K)$ norm:

# In[15]:


# compute and print errors
max_wnd_error = max(wnd_error)
l2_wnd_error = np.sqrt(K.integrate_over_boundary(wnd_error ** 2))
print('Max pointwise error = %.4e'%max_wnd_error)
print('L^2 norm of wnd error = %.4e'%l2_wnd_error)


# ### Find an Anti-Laplacian
# 
# Our second task is to find an anti-Laplacian $\Phi$ such that $\Delta\Phi=\phi$.
# Note that
# \begin{align*}
# 	\Lambda(x) = \frac14 |x|^2 \big(\ln|x|-1 \big)
# \end{align*}
# is an anti-Laplacian of $\lambda(x) = \ln|x|$.
# So if $\Psi$ is an anti-Laplacian of $\psi$, then we would have
# \begin{align*}
# 	\Phi(x) = \Psi(x) + \sum_{k=1}^m a_k \Lambda(x-\xi_k)
# \end{align*}
# is an anti-Laplacian of 
# \begin{align*}
# 	\phi(x) = \psi(x) + \sum_{k=1}^m a_k \ln|x-\xi_k|
# 	~.
# \end{align*}
# All of this handled internally when we call the 
# `compute_anti_laplacian_harmonic_part()` method.

# In[16]:


v.compute_anti_laplacian_harmonic_part()


# Let's compare the computed values of $\Phi$ to 
# \begin{align*}
# 	\tilde\Phi(x) = \frac14 e^{x_1}
# 	\big(x_1 \cos x_2 + x_2 \sin x_2\big)
# 	+ \frac14 |x - \xi|^2 \big(\ln|x - \xi|-1 \big)
# 	~,
# \end{align*}
# which is an anti-Laplacian of $\phi$.

# In[17]:


# an exact anti-Laplacian
PHI_exact = 0.25 * np.exp(x1) * (x1 * np.cos(x2) + x2 * np.sin(x2)) + \
	a_exact * 0.25 * ((x1 - xi[0]) ** 2 + (x2 - xi[1]) ** 2) * ( \
    0.5 * np.log((x1 - xi[0]) ** 2 + (x2 - xi[1]) ** 2) - 1)

# computed anti-Laplacian
PHI_computed = v.get_anti_laplacian_harmonic_part()

quad_list = [q_trap, q_kress,]
f_trace_list = [PHI_exact, PHI_computed]
fmt = ('g--', 'k.')
legend = ('exact','computed')
title = 'Anti-Laplacian'
pf.plot_trace(f_trace_list, fmt, legend, title, K, quad_list)


# In general, $\Phi$ is unique only up to the addition of a harmonic function.
# Indeed, if $\Phi$ and $\widetilde\Phi$ are both anti-Laplacians of $\phi$, 
# we have 
# \begin{align*}
# 	\Delta (\Phi - \widetilde\Phi) = \phi - \phi = 0
# 	~.
# \end{align*}
# However, in this case we can say more. 
# As a consequence of the way that $\Phi$ and $\widetilde\Phi$ were computed, 
# it ought to hold that
# \begin{align*}
# 	\Psi - \widetilde\Psi = c \cdot x
# \end{align*}
# is a linear function.
# We will test this conjecture by performing a least squares best linear fit 
# on the computed values of $\Psi - \widetilde\Psi$.

# In[18]:


PHI_diff = PHI_exact - PHI_computed

X = np.zeros((K.num_pts, 2))
X[:,0] = x1
X[:,1] = x2
XX = np.transpose(X) @ X
Xy = np.transpose(X) @ PHI_diff
aa = np.linalg.solve(XX, Xy)
PHI_diff_fit = X @ aa

PHI_diff_error = np.abs(PHI_diff_fit - PHI_diff)

quad_list = [q_trap, q_kress,]
f_trace_list = [PHI_diff_fit, PHI_diff]
fmt = ('b--','k.')
legend = ('least squares best linear fit', 'exact - computed')
title = 'Anti-Laplacian difference'
pf.plot_trace(f_trace_list, fmt, legend, title, K, quad_list)

quad_list = [q_trap, q_kress,]
f_trace_list = [PHI_diff_error,]
fmt = ('k.',)
legend = ('exact - computed - linear fit',)
title = 'Anti-Laplacian error'
pf.plot_trace_log(f_trace_list, fmt, legend, title, K, quad_list)


# As before, let's compute the maximum pointwise error and the 
# $L^2(\partial K)$ error.

# In[19]:


max_PHI_error = max(PHI_diff_error)
l2_PHI_error = np.sqrt(K.integrate_over_boundary(PHI_diff_error ** 2))
print('Max pointwise error = %.4e'%max_PHI_error)
print('L^2 norm of error = %.4e'%l2_PHI_error)


# Before we use $v$ for computations, we need to compute the trace and 
# weighted normal derivative of $P$, the polynomial part of $v = \phi + P$.

# In[20]:


v.compute_polynomial_part_trace()
v.compute_polynomial_part_weighted_normal_derivative()


# ## Define another function
# 
# Let $w : K \to \mathbb{R}$ is given by 
# \begin{align*}
# 	w(x) = \frac{x_1 - 0.5}{(x_1 - 0.5)^2 + (y - 0.5)^2} + x_1^3 + x_1 x_2^2
# 	~.
# \end{align*}
# Again, we have that $w = \psi + Q$ is the sum of a harmonic function and 
# a polynomial, with
# \begin{align*}
# 	\psi(x) = \frac{x_1 - 0.5}{(x_1 - 0.5)^2 + (y - 0.5)^2} 
# 	~,
# 	\quad
# 	Q(x) = x_1^3 + x_1 x_2^2
# 	~.
# \end{align*}
# (The notation "$\psi$" is being recycled here.)

# In[21]:


# trace of w
w_trace = (x1 - 0.5) / ((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2) + \
    x1 ** 3 + x1 * x2 ** 2

# define a monomial term by specifying its multi-index and coefficient
w_laplacian = pf.polynomial([ [8.0, 1, 0] ])

# declare w as local function object
w = pf.locfun(solver, lap_poly=w_laplacian, has_poly_trace=False)
w.set_trace_values(w_trace)


# For convenience, we don't need to call all of the `compute` methods we did for
# $v$. Instead, we call `compute_all()`.

# In[22]:


w.compute_all()


# ## $H^1$ semi-inner product
# 
# We are now ready to compute the $H^1$ semi-inner product between $v$ and $w$.
# This can be done by calling the `compute_h1()` method from either function.

# In[23]:


h1_vw_computed = v.get_h1_semi_inner_prod(w)
print('H^1 semi-inner product (vw) = ', h1_vw_computed)


# In exact arithmetic, the $H^1$ semi-inner product is symmetric.
# Let's check that we get the same thing if we compute in the opposite order.

# In[24]:


h1_wv_computed = w.get_h1_semi_inner_prod(v)
print('H^1 semi-inner product (wv) = ', h1_wv_computed)


# Here's the difference between the two:

# In[25]:


print('Difference in computed H^1 = ', abs(h1_vw_computed - h1_wv_computed))


# Finally, let's compare this to the value obtained with *Mathematica*:
# \begin{align*}
# 	\int_K \nabla v \cdot \nabla w ~ dx
# 	&\approx 4.46481780319135
# 	\pm 9.9241 \times 10^{-15}
# \end{align*}
# where the value after "$\pm$" indicates the estimated error in this value
# according to *Mathematica*.

# In[26]:


h1_vw_exact = 4.46481780319135
print('H^1 error (vw) = ', abs(h1_vw_computed - h1_vw_exact))
print('H^1 error (wv) = ', abs(h1_wv_computed - h1_vw_exact))


# ## $L^2$ Inner Product
# 
# Let's compute the $L^2$ inner product
# \begin{align*}
# 	\int_K v \, w ~dx
# 	&\approx 1.39484950156676
# 	\pm 2.7256 \times 10^{-16}
# \end{align*}
# whose approximate value was obtained with *Mathematica*.

# In[27]:


l2_vw_computed = v.get_l2_inner_prod(w)
print('L^2 inner product (vw) = ',l2_vw_computed)
l2_wv_computed = w.get_l2_inner_prod(v)
print('L^2 inner product (wv) = ', l2_wv_computed)
print('Difference in computed L^2 = ', abs(l2_vw_computed - l2_wv_computed))
l2_vw_exact = 1.39484950156676
print('L^2 error (vw) = ', abs(l2_vw_computed - l2_vw_exact))
print('L^2 error (wv) = ', abs(l2_wv_computed - l2_vw_exact))


# ## Convergence Studies
# 
# We repeated the above experiment for several values of the quadrature parameter
# $n$ (where each edge of $\partial K$ is sampled at $2n$ points).
# 
# Here's what we found for the intermediate computations on $v$:
# 
# |	n	|	a_1 error	|	hc error	|	wnd error	|	al error	|	H1 error	|	L2 error	|
# |-------|---------------|---------------|---------------|---------------|---------------|---------------|
# |	4	|	1.7045e-03	|	3.5785e-02	|	2.8201e-01	|	8.3234e-03	|	1.5180e-02	|	3.4040e-03	|
# |	8	|	3.5531e-07	|	2.6597e-04	|	1.2855e-03	|	3.9429e-05	|	2.6758e-04	|	8.3812e-05	|
# |	16	|	1.0027e-09	|	1.1884e-06	|	3.7415e-06	|	3.3785e-07	| 	8.4860e-07	|	3.8993e-08	|
# |	32	|	3.5905e-13	|	2.3095e-09	|	1.0434e-08	|	1.9430e-09	|	1.0860e-09	|	2.8398e-11	|
# |	64	|	1.8874e-14	|	1.6313e-12	|	6.4780e-11	|	7.0728e-12	|	9.5390e-13	|	1.1036e-13	|

# In[28]:


print(q_kress.n)

print('')

print('log coef error = %.4e'%abs(v.log_coef[0] - a_exact))
print('L^2 norm of hc error = %.4e'%l2_hc_error)
print('L^2 norm of wnd error = %.4e'%l2_wnd_error)
print('L^2 norm of antilap error = %.4e'%l2_PHI_error)

print('')

print('H^1 error (vw) = %.4e'%abs(h1_vw_computed - h1_vw_exact))
print('L^2 error (vw) = %.4e'%abs(l2_vw_computed - l2_vw_exact))


# ## Bonus: Find Interior Values
# 
# Cauchy's integral formula
# \begin{align*}
# 	f(z) = \frac{1}{2\pi} \oint_{\partial K} \frac{f(\zeta)}{\zeta -z}d\zeta~,
# 	\quad z = x_1 + \mathfrak{i}x_2 \mapsto x = (x_1, x_2)\in K
# \end{align*}
# holds for multiply connected domains, provided that the outer boundary 
# is oriented counterclockwise and the inner boundaries or oriented clockwise.
# (Proof left as an exercise.)
# We can use this to evaluate $f = \psi + \mathfrak{i}\widehat\psi$,
# and thereby determine the interior values of $v$. 

# In[29]:


y1 = K.int_x1
y2 = K.int_x2

v.compute_interior_values()

v_computed = v.int_vals
v_x1_computed = v.int_grad1
v_x2_computed = v.int_grad2

plt.figure()
plt.contourf(y1, y2, v_computed, levels=50)
plt.colorbar()
plt.title('Interior values of $v$')
plt.show()


# Since we have an explicit formula for $v$ in the interior,
# we can plot the pointwise errors.
# (Note that the scale is logarithmic.)

# In[30]:


v_exact = np.exp(y1) * np.cos(y2) + \
    0.5 * a_exact * np.log((y1 - xi[0]) ** 2 + (y2 - xi[1]) ** 2) + \
    y1 ** 3 * y2 + y1 * y2 ** 3

v_error = np.log10(np.abs(v_computed - v_exact))

plt.figure()
plt.contourf(y1, y2, v_error, levels=50)
plt.colorbar()
plt.title('Interior errors ($\log_{10}$)')
plt.show()


# Let's do the same for the components of the gradient:

# In[31]:


v_x1_exact = np.exp(y1) * np.cos(y2) + \
    a_exact * (y1 - xi[0]) / ((y1 - xi[0]) ** 2 + (y2 - xi[1]) ** 2) + \
    3 * y1 ** 2 * y2 + y2 ** 3

v_x1_error = np.log10(np.abs(v_x1_computed - v_x1_exact))

plt.figure()
plt.contourf(y1, y2, v_x1_error, levels=50)
plt.colorbar()
plt.title('Gradient errors in $x_1$ ($\log_{10}$)')
plt.show()


# In[32]:


v_x2_exact = -np.exp(y1) * np.sin(y2) + \
    a_exact * (y2 - xi[1]) / ((y1 - xi[0]) ** 2 + (y2 - xi[1]) ** 2) + \
    y1 ** 3 + 3 * y1 * y2 ** 2

v_x2_error = np.log10(np.abs(v_x2_computed - v_x2_exact))

plt.figure()
plt.contourf(y1, y2, v_x2_error, levels=50)
plt.colorbar()
plt.title('Gradient errors in $x_2$ ($\log_{10}$)')
plt.show()

