# Tutorial Guide

These tutorials are designed to help users of the `puncturedfem` package become acquainted with some of its features, and how to avoid some potential pitfalls. 

**Note:** The tutorials are still under construction; be sure to check back in the future!

The tutorials are partitioned into four chapters, each with a different focus.

## Chapter 1. Mesh Geometry
This chapter is devoted to building curvilinear and punctured meshes, which is somewhat more involved than conventional triangulations.

- [1.1: Vertices and Edges](1.1-vertices-edges.ipynb)
- [1.2: Mesh Construction](1.2-meshes.ipynb)

## Chapter 2. Local Poisson Spaces
This chapter explores the *local Poisson space* $V_p(K)$, which will be used to construct a finite element problem in Chapter 3.

- [2.1: Polynomials](2.1-polynomials.ipynb)
- [2.2: Dirichlet Traces](2.2-traces.ipynb)
- [2.3: Local Poisson Functions](2.3-local-functions.ipynb)
- [2.4: Local Poisson Spaces](2.4-local-spaces.ipynb)

## Chapter 3. Finite Elements with Poisson Spaces
This chapter introduces global Poisson spaces, and how they can be used in finite element methods.

- [3.1: Global Poisson Spaces](3.1-global-spaces.ipynb)
- [3.2: Global Boundary Conditions](3.2-global-bc.ipynb)

## Chapter 4.Advanced Features
This chapter covers some features that not all users are likely to need, but may be important in some situations.

- [4.1: Nyström Solvers](4.1-nystrom.ipynb)
- [4.2: Heavy Sampling of an Intricate Edge](4.2-heavy-sampling.ipynb)