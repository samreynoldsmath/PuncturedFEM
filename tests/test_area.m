%%
clear
close all
format compact
format short g

%% finite element formulation for landscape function
% B = @(Amat,Mmat) Amat+Mmat; % -\Delta u + u = f
% f = @(x,y) 1; % weak formulation: B(u,v) = l(f,v)

%% mesh cell parameters
% args.id = 'square';
args.id = 'square-circular-hole';
args.r_in = 0.25;
args.inner_square_scale = 0.35;
args.num_inner_edges = 2;

%% global mesh parameters
args.r = 16;          % r x r mesh on the unit square 
args.deg = 1;        % polynomial degree, V_m(K)

%% solver parameters
args.type = 'density';
args.restart = 10;
args.tol = 1e-12;
args.maxit = 10;

%% quadrature parameters
args.n = 4;         % 2n points sampled per edge
args.kress_sig = 7;  % Kress quadrature parameter \sigma > 2

%% include dependencies
addpath('classes','builders','PuncturedDomainHigherOrderQuadrature')

%% build quadrature
quad = BuildQuads(args.n,args.kress_sig);

%% build element K
elm = BuildElement(args,quad);

disp(elm)
figure(1),clf,elm.plot_boundary('quiver')

%%
f = ones(elm.num_pts,1);
p = class_poly();
p = p.zero_poly();

g = f;
q = p;

z = [0,0];

%%
[L2,H1]=PuncturedHighOrderQuadrature(elm,quad,z,p,f,q,g,args)


