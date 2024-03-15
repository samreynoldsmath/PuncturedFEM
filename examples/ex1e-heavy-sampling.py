#!/usr/bin/env python
# coding: utf-8

# # Example 1.E: Heavy Sampling of an Intricate Edge
# ### Sam Reynolds, 2024
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
        freq=4,
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
nyst = pf.NystromSolver(K_simple, debug=True)

# plot boundary
pf.plot.MeshPlot(K_simple.get_edges()).draw()


# It is simple to verify that $v\in V_1(K)$ given by $v(x_1,x_2) = x_2$ has a square $L^2$ norm of
# \begin{align*}
#     \int_K v^2 ~dx = \frac13~.
# \end{align*}
# Let's verify this:

# In[ ]:


# define v to have a Dirichlet trace of x_2 on each edge
x2 = pf.Polynomial([(1.0, 0, 1)])
v_trace = pf.PiecewisePolynomial(num_polys=4, polys=[x2, x2, x2, x2])

# the local function v = x_2 is harmonic
v = pf.LocalFunction(nyst=nyst, lap_poly=pf.Polynomial(), poly_trace=v_trace)
v.compute_all()

# compute area and error
L2_exact = 1 / 3
L2_computed = v.get_l2_inner_prod(v)
print(f"Error = {abs(L2_exact - L2_computed):.4e}")


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
    freq=32,  # increase frequency
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
nyst = pf.NystromSolver(K, debug=True)

# the harmonic function v = x_2
v = pf.LocalFunction(nyst=nyst, lap_poly=pf.Polynomial(), poly_trace=v_trace)
v.compute_all()

# compute square L^2 norm and error
L2_exact = 1 / 3
L2_computed = v.get_l2_inner_prod(v)
print(f"Error = {abs(L2_exact - L2_computed):.4e}")


# One might expect that if we increase the sampling parameter, this error will get smaller. 
# However, we soon discover that this crashes the `NystromSolver` initialization.

# In[ ]:


# get 1024 sampled points on each edge
K.parameterize(quad_dict=pf.get_quad_dict(n=512))

try:
    # (WARNING!) this line will result in an exception being thrown
    nyst = pf.NystromSolver(K, debug=True)
except ZeroDivisionError as e:
    print("Indeed, an exception was thrown!\n", e)


# ## Changing the Kress parameter (optional)
# As we saw in [Example 0](ex0-mesh-building.ipynb), we can change the Kress parameter $p$ to adjust how much the sampled points are "clustered" near the endpoints. 
# The default value is $p=7$, but changing this to its lowest value $p=2$ results in sampled points that are more spread out, perhaps enough so that we can avoid division by machine zero.
# 
# **NOTE:** The condition number of the Nyström matrix is very high and GMRES will not converge quickly, if at all. Uncomment the following cell to see this.

# In[ ]:


# # get 1024 sampled points on each edge with lower Kress parameter
# K.parameterize(quad_dict=pf.get_quad_dict(n=512, p=2))
# nyst = pf.NystromSolver(K, debug=True)

# # the harmonic function v = x_2
# v = pf.LocalFunction(nyst=nyst, lap_poly=pf.Polynomial(), poly_trace=v_trace)

# # (WARNING!) this line will take a long time to run
# v.compute_all()

# # compute square L^2 norm and error
# L2_exact = 1 / 3
# L2_computed = v.get_l2_inner_prod(v)
# print(f"Error = {abs(L2_exact - L2_computed):.4e}")


# ## Splitting Edges
# 
# As we saw in [Example 0](ex0-mesh-building.ipynb), we can split edges in two using the `split_edge()` function. Let's try splitting the 'bad' edge into smaller edges.

# In[ ]:


# replace edge 0 with eight new edges
edges += pf.split_edge(edges[0], num_edges=8)
del edges[0]

# define mesh cell
K = pf.MeshCell(idx=0, edges=edges)


# In the previous section, we tried sampling each edge with $2n = 1024$ points. Notice, though, that only the bottom edge is problematic, and we might get away with sampling the straight edges at a lower rate. To keep the number of sampled points on the bottom edge the same, which has now been split into 8 edges, we need to set the sampling parameter to $n=64=512/8$.

# In[ ]:


# bottom edge sampled at 1024 points
K.parameterize(quad_dict=pf.get_quad_dict(n=64))

# set up Nystrom solver
nyst = pf.NystromSolver(K, debug=True)


# The `NystromSolver` didn't crash this time. Let's define the local function $v = x_2$ and take a peek at its trace:

# In[ ]:


# Dirichlet trace of the harmonic function v = x_2
x2 = pf.Polynomial([(1.0, 0, 1)])
v_trace = pf.DirichletTrace(K, funcs=x2)

# the harmonic function v = x_2
v = pf.LocalFunction(nyst=nyst, lap_poly=pf.Polynomial(), has_poly_trace=False)
v.set_trace_values(v_trace.values) # TODO: pass DirichletTrace object directly to LocalFunction

# plot the trace of v
# TODO: TracePlot should be able to initialize its own quad_dict
pf.plot.TracePlot(v_trace.values, K, quad_dict=pf.get_quad_dict(n=64)).draw()


# Finally, let's see if we can accurately compute our quantity of interest:

# In[ ]:


# this can take a few minutes
v.compute_all()
L2_exact = 1 / 3
L2_computed = v.get_l2_inner_prod(v)
print(f"Error = {abs(L2_exact - L2_computed):.4e}")

