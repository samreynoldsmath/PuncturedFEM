%%
clear
close all
format compact
format short e
addpath(...
    'classes',...
    'builders',...
    'PuncturedDomainHigherOrderQuadrature')

%% mesh cell parameters
% dargs.id = 'square';

dom_args.id = 'square-circular-hole';
dom_args.r_in = 0.25;
dom_args.num_inner_edges = 2;

% dargs.id = 'square-square-hole';
% dargs.inner_square_scale = 0.35;

%% global mesh parameters
mesh_args.r = 4;          % r x r mesh on the unit square 

%% function space parameters
funsp_args.deg = 1; % polynomial degree, V_m(K)

%% Dirichlet-to-Neumann map parameters
d2n_args.type = 'density';
d2n_args.restart = 10;
d2n_args.tol = 1e-8;
d2n_args.maxit = 5;

%% quadrature parameters
quad_args.n = 64;         % 2n points sampled per edge
quad_args.kress_sig = 7;  % Kress quadrature parameter \sigma > 2

%% local interior value parameters (for plotting)
iargs.n_x = 64;
iargs.n_y = iargs.n_x;
iargs.tol = 0.01;
iargs.num_funcs = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% build quadrature
quad = BuildQuads(quad_args);
for q = 1:3
    disp(quad{q})
end

%% build element K
elm = BuildElement(dom_args,quad);
disp(elm)

%% harmonic function and a harmonic conjugate (of non-logartimic part)
u = @(x,y) x-y+0.5*log((x-0.5).^2+(y-0.5).^2);
% u = @(x,y) x-y;
v = @(x,y) x+y-1;

%% traces
f = u(elm.x(:,1),elm.x(:,2));
v_exact = v(elm.x(:,1),elm.x(:,2));

%%
intval = class_interior_values;
intval = intval.init(iargs);
intval = intval.get_physical_pts(elm);

intval.val = intval.cauchy_integral_harmonic_function(f,elm,quad,d2n_args);
intval.val = intval.val(:);

figure(3),clf
fig_type = 'contour';
intval.plot_v(1,elm,fig_type)

%%
X = intval.x;
Y = intval.y;
Z = u(X,Y);
Z(~intval.in) = NaN;

figure(4),clf
elm.plot_boundary('basic-')
hold on
contour(X,Y,Z,50)
hold off

%%
figure(5),clf
ZZ = reshape(intval.val,intval.n_x,intval.n_y);
E = log10(abs(Z-ZZ));
elm.plot_boundary('basic-')
hold on
contour(X,Y,E,50)
colorbar
hold off

