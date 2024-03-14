#!/usr/bin/env python
# coding: utf-8

# # Example 1.E: Heavy Sampling of an Intricate Edge
# 
# We may sometimes have an edge that has fine details that need to be resolved by increasing the sampling parameter $n$, with the edge being sampled at $2n+1$ points, including the end points.
# 
# ## When Things Go Right
# 
# For example, consider a unit square with one of the edges being sinusoidal.

# In[ ]:


import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import puncturedfem as pf
import numpy as np
import matplotlib.pyplot as plt

# define vertices
verts: list[pf.Vert] = []
verts.append(pf.Vert(x=0.0, y=0.0))
verts.append(pf.Vert(x=1.0, y=0.0))
verts.append(pf.Vert(x=1.0, y=1.0))
verts.append(pf.Vert(x=0.0, y=1.0))

# define edges
edges: list[pf.Edge] = []
edges.append(
    pf.Edge(
        verts[0],
        verts[1],
        pos_cell_idx=0,
        curve_type="sine_wave",
        amp=0.1,
        freq=8,
    )
)
edges.append(pf.Edge(verts[1], verts[2], pos_cell_idx=0))
edges.append(pf.Edge(verts[2], verts[3], pos_cell_idx=0))
edges.append(pf.Edge(verts[3], verts[0], pos_cell_idx=0))

# define mesh cell
K_simple = pf.MeshCell(idx=0, edges=edges)

# parameterize edges
K_simple.parameterize(quad_dict=pf.get_quad_dict(n=64))

# set up Nystrom solver
nyst = pf.NystromSolver(K_simple)

# plot boundary
pf.plot.MeshPlot(K_simple.get_edges()).draw()


# Notice that the area of the mesh cell $K$ is $|K|=1$.
# We could compute this by integrating the constant function $v=1$:
# \begin{align*}
#     1 = \int_K v^2~dx~.
# \end{align*}

# In[ ]:


A_op = nyst.double_layer_op
I = np.eye(A_op.shape[0])
A = np.zeros((A_op.shape[0], A_op.shape[1]))
for i in range(A_op.shape[0]):
    A[:, i] = A_op @ I[:, i]
A += np.ones(A.shape) * 2 * np.pi * K_simple.num_edges / K_simple.num_pts
print(f"Condition number of A = {np.linalg.cond(A):.2e}")


# In[ ]:


# define v to have a Dirichlet trace of 1 on each edge
one = pf.Polynomial([(1.0, 0, 0)])
v_trace = pf.PiecewisePolynomial(num_polys=4, polys=[one, one, one, one])

# the constant function v = 1 is harmonic
v = pf.LocalFunction(nyst=nyst, lap_poly=pf.Polynomial(), poly_trace=v_trace)
v.compute_all()

# compute area and error
area_exact = 1.0
area_computed = v.get_l2_inner_prod(v)
print(f"Error in computed area = {np.abs(area_exact - area_computed)}")


# ## When Things Go Wrong
# 
# Let's make this example more interesting by increasing the frequency of the sinusoid on the bottom of the square.

# In[ ]:


# crazy edge
edges[0] = pf.Edge(
    verts[0],
    verts[1],
    pos_cell_idx=0,
    curve_type="sine_wave",
    amp=0.1,
    freq=8,  # this is scary
)

# define and parameterize a new mesh cell
K = pf.MeshCell(idx=0, edges=edges)
K.parameterize(quad_dict=pf.get_quad_dict(n=64))

# and look at it
pf.plot.MeshPlot(K.get_edges()).draw()


# That doesn't look right... 
# We can change the sampling parameter $n$ when initializing a `MeshPlot` instance to get more resolution. We also need to set the `reparameterize` flag to `True`.

# In[ ]:


pf.plot.MeshPlot(K.get_edges(), reparameterize=True, n=512).draw()


# That looks pretty good, but note that `MeshPlot` didn't overwrite the sampled points we got above with `n=64`:

# In[ ]:


print(f"n = {K.num_pts // K.num_edges // 2}")


# Since this is not a high enough sampling rate to capture the high frequency of the bottom edge, we might expect our computation of the area to not be very accurate.
# Let's confirm this suspicion:

# In[ ]:


# set up Nystrom solver
nyst = pf.NystromSolver(K, verbose=True)

# the constant function v = 1
v = pf.LocalFunction(nyst=nyst, lap_poly=pf.Polynomial(), poly_trace=v_trace)
v.compute_all()

# compute area and error
area_exact = 1.0
area_computed = v.get_l2_inner_prod(v)
print(f"Error in computed area = {np.abs(area_exact - area_computed)}")


# One might expect that if we increase the sampling parameter, this error will get smaller. 
# However, we soon discover that this crashes the `NystromSolver` initialization.

# In[ ]:


# get 1024 sampled points on each edge
K.parameterize(quad_dict=pf.get_quad_dict(n=512))

# (WARNING!) this line will result in an exception being thrown
nyst = pf.NystromSolver(K, verbose=True)


# ## Changing the Kress parameter (optional)
# As we saw in [Example 0](ex0-mesh-building.ipynb), we can change the Kress parameter $p$ to adjust how much the sampled points are "clustered" near the endpoints. 
# The default value is $p=7$, but changing this to its lowest value $p=2$ results in sampled points that are more spread out, perhaps enough so that we can avoid division by machine zero. Let's try it (this may take a while):

# In[ ]:


# get 1024 sampled points on each edge with lower Kress parameter
K.parameterize(quad_dict=pf.get_quad_dict(n=512, p=2))
nyst = pf.NystromSolver(K, verbose=True)


# The `NystromSolver` initialized without errors, so let's try to compute our quantity of interest:

# In[ ]:


# the constant function v = 1
v = pf.LocalFunction(nyst=nyst, lap_poly=pf.Polynomial(), poly_trace=v_trace)
v.compute_all()

# compute area and error
area_exact = 1.0
area_computed = v.get_l2_inner_prod(v)
print(f"Error in computed area = {np.abs(area_exact - area_computed)}")


# This worked well enough, but $10^{-6}$ is kind of a big error for the amount of work we had to do. Let's see if we can do better with a different strategy.

# ## Splitting Edges
# 
# As we saw in [Example 0](ex0-mesh-building.ipynb), we can split edges in two using the `split_edge()` function. Let's try splitting the 'bad' edge into smaller edges.

# In[ ]:


# split edge 0 in half
e1, e2 = pf.split_edge(e=edges[0], t_split=np.pi)

# split into quarters
e1_1, e1_2 = pf.split_edge(e1, t_split=np.pi / 2)
e2_1, e2_2 = pf.split_edge(e2, t_split=3 * np.pi / 2)

# split into eighths
e1_1_1, e1_1_2 = pf.split_edge(e1_1, t_split=np.pi / 4)
e1_2_1, e1_2_2 = pf.split_edge(e1_2, t_split=3 * np.pi / 4)
e2_1_1, e2_1_2 = pf.split_edge(e2_1, t_split=5 * np.pi / 4)
e2_2_1, e2_2_2 = pf.split_edge(e2_2, t_split=7 * np.pi / 4)

# replace edge 0 with eight new edges
edges += [e1_1_1, e1_1_2, e1_2_1, e1_2_2, e2_1_1, e2_1_2, e2_2_1, e2_2_2]
del edges[0]

# define mesh cell
K = pf.MeshCell(idx=0, edges=edges)

# bottom edge sampled at 1024 points
K.parameterize(quad_dict=pf.get_quad_dict(n=64))

# set up Nystrom solver
nyst = pf.NystromSolver(K, verbose=True)


# In[ ]:


A_op = nyst.double_layer_op
I = np.eye(A_op.shape[0])
A = np.zeros((A_op.shape[0], A_op.shape[1]))
for i in range(A_op.shape[0]):
    A[:, i] = A_op @ I[:, i]
A += np.ones(A.shape) * 2 * np.pi * K.num_edges / K.num_pts
print(f"Condition number of A = {np.linalg.cond(A):.2e}")


# The `NystromSolver` didn't crash this time. 
# Let's see if we can accurately compute the area:

# In[ ]:


# Dirichlet trace of constant function v = 1
v_trace = pf.PiecewisePolynomial(
    num_polys=K.num_edges, polys=K.num_edges * [one]
)

# constant function v = 1
v = pf.LocalFunction(nyst=nyst, lap_poly=pf.Polynomial(), poly_trace=v_trace)
v.compute_all()

# compute area and error
area_exact = 1.0
area_computed = v.get_l2_inner_prod(v)
print(f"Error in computed area = {np.abs(area_exact-area_computed)}")


# This gave us a much better error than changing the Kress parameter, and the computation was much faster.

# ## Using the DirchletTrace class on MeshCells with Split Edges
# *Note:* The technique used in this section is likely to be automated in future releases.
# 
# Consider the harmonic function $v \in V_1(K)$ with a linear trace 
# \begin{align*}
#     v|_{\partial K}(x,y) = y
# \end{align*}
# on the bottom edge of the square, and zero on the other three edges.
# We will use the `DirichletTrace` class to define the corresponding `LocalFunction` object.
# 
# The `DirichletTrace` class is initialized with a list of `Edge` objects, or, alternatively, a `MeshCell` object, on which the trace is to be defined.
# We have several ways to set the trace values. 
# - We can define a function $f : \mathbb{R}^2 \to \mathbb{R}$ and pass this function to the `set_funcs()` method. **Note:** The function $f$ must be typed, e.g. `def f(x: float, y: float) -> float: return x*y`. Untyped functions will throw an exception. The types can be `float`, `int`, or `numpy.ndarray`.
# - We can set the trace function on a specific edge with the `set_funcs_on_edge()` method, which accepts the index of the edge (i.e. its position in the list of edges) and a function $f$ as above.
# - We can set the trace values directly with the `set_values()` method, which accepts a `numpy.ndarray` of shape `(N,1)`, where `N` is the number of points on the edge.  We can set the trace to a constant by passing a `float` or `int` instead of an array.
# - We can set the trace values on a single edge with the `set_values_on_edge()` method, which accepts a `numpy.ndarray` of shape `(2n,1)`, where `2n` is the number of points on the edge, excluding the terminal point. Again, we can set the trace to a constant by passing a `float` or `int` instead of an array.
# 
# Below, we use the `set_funcs()` method to define the same trace function on all the edges.

# In[ ]:


v_trace = pf.DirichletTrace(K, custom=True)
def f(x: float, y: float) -> float:
    return y
v_trace.set_funcs(f)
v_trace.find_values()


# We can set the trace values on the three straight edges (which have indices 0, 1,  and 2) to zero by using the `set_values_on_edge()` method.

# In[ ]:


for k in [0, 1, 2]:
    v_trace.set_trace_values_on_edge(k, 0)


# Let's take a look at a plot of the trace values:

# In[ ]:


pf.plot.TracePlot(v_trace.values, K, quad_dict=pf.get_quad_dict(n=64)).draw()


# Let's define the `LocalFunction` object and compute the harmonic conjugate, normal derivative, etc.

# In[ ]:


v = pf.LocalFunction(nyst=nyst, lap_poly=pf.Polynomial(), has_poly_trace=False)
v.set_trace_values(v_trace.values)
v.compute_all()


# The `compute_all()` method does not compute interior values, so let's do that now and take a look at the plot:

# In[ ]:


v.compute_interior_values()
pf.plot.LocalFunctionPlot(v).draw()


# ## TODO: Creating a LocalFunctionSpace on a MeshCell with a Split Edge
# This is a [planned feature](../doc/TODO.md).
