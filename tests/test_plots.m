%%
clear
close all
format compact
format short e
addpath('classes','builders','PuncturedDomainHigherOrderQuadrature')

%% mesh cell parameters
% dargs.id = 'square';

dargs.id = 'square-circular-hole';
dargs.r_in = 0.25;
dargs.num_inner_edges = 2;

% dargs.id = 'square-square-hole';
% dargs.inner_square_scale = 0.35;

%% global mesh parameters
margs.r = 4;          % r x r mesh on the unit square 

%% function space parameters
vargs.deg = 3; % polynomial degree, V_m(K)

%% Dirichlet-to-Neumann map parameters
D2N_args.type = 'density';
D2N_args.restart = 10;
D2N_args.tol = 1e-6;
D2N_args.maxit = 5;

%% quadrature parameters
qargs.n = 64;         % 2n points sampled per edge
qargs.kress_sig = 7;  % Kress quadrature parameter \sigma > 2

%% local interior value parameters (for plotting)
iargs.n_x = 64;
iargs.n_y = iargs.n_x;
iargs.tol = 0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% build quadrature
quad = BuildQuads(qargs);
disp(quad{2})

%% build element K
elm = BuildElement(dargs,quad);
disp(elm)
figure(1),clf,elm.plot_boundary('quiver')

%% construct mesh 
mesh = class_uniform_mesh_square_domain;
mesh = mesh.init(elm,margs);

disp(mesh)
figure(2),clf,mesh.draw(elm)

%% generate a basis of V_m(K)
vmk = class_local_function_space;
vmk = vmk.init(elm,quad,vargs);

disp(vmk)
vmk.print(101,elm,quad)

%% set up local interior grid points (for basis function plots)
iargs.num_funcs = vmk.dim;
intval = class_interior_values;
intval = intval.init(elm,iargs);

disp(intval)
figure(3),clf,intval.show_points(elm);


%% get interior values of basis functions, make contour plots
intval = intval.get_int_vals(vmk,elm,quad,D2N_args);

fig_type = 'minimal';
intval.plot_all(201,elm,fig_type)

