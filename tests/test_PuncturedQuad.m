%%
clear
close all
format compact
format short e
addpath('classes','builders','PuncturedDomainHigherOrderQuadrature')

%% 
do_L2 = false;
do_H1 = true;

%% mesh cell parameters
dargs.id = 'annulus';
dargs.r_out = 1;
dargs.r_in = 0.25;
dargs.num_inner_edges = 1;
dargs.num_outer_edges = 1;
dargs.num_edges = dargs.num_inner_edges+dargs.num_outer_edges;

% dargs.id = 'square';

% dargs.id = 'square-circular-hole';
% dargs.r_in = 0.25;

%% Dirichlet-to-Neumann map parameters
D2N_args.type = 'density';
D2N_args.restart = 10;
D2N_args.tol = 1e-8;
D2N_args.maxit = 5;

%% quadrature parameters
qargs.n = 8;         % 2n points sampled per edge
qargs.kress_sig = 7;  % Kress quadrature parameter \sigma > 2

%% build quadrature
quad = BuildQuads(qargs);

%% build element K
elm = BuildElement(dargs,quad);

%% define v
shift = [0,0];
x = elm.x(:,1);
y = elm.x(:,2);

% trace
% f = 0.5*log((x-shift(1)).^2+(y-shift(2)).^2);
% f = x-y+x.*y+0.5*log(x.*x+y.*y);
% f = x-y+x.*y;
% f = x.*y;
% f = x./(x.^2+y.^2);
% f = (exp(x).*cos(y)-1)./( 1+exp(2*x)-2*exp(x).*cos(y) );

f = zeros(elm.num_pts,1);
f(elm.get_seg_point_idx(2)) = 1;

% f = -0.25*log((x-shift(1)).^2+(y-shift(2)).^2)/log(2);
% f = 1+0.25*log((x-shift(1)).^2+(y-shift(2)).^2)/log(2);

% polynomial Laplacian
p = class_poly;
p = p.zero_poly();

%% define w
g = f;
q = p;

%% shift for monomial basis
z = [0,0];

%% compute integrals
[L2,H1] = InnerProducts(f,p,g,q,z,elm,quad,D2N_args,do_L2,do_H1)







