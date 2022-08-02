%%
clear
close all
format compact
format short e
addpath('classes','builders','PuncturedDomainHigherOrderQuadrature')

%% mesh cell parameters
% dargs.id = 'circle';
% dargs.r = 1.23;
% dargs.num_edges = 2;

dargs.id = 'annulus';
dargs.r_out = 0.456;
dargs.r_in = 0.123;
dargs.num_inner_edges = 1;
dargs.num_outer_edges = 2;
dargs.num_edges = dargs.num_inner_edges+dargs.num_outer_edges;

% dargs.id = 'square';

% dargs.id = 'square-circular-hole';
% dargs.r_in = 0.25;
% dargs.num_inner_edges = 2;

%% Dirichlet-to-Neumann map parameters
D2N_args.type = 'density';
D2N_args.restart = 10;
D2N_args.tol = 1e-12;
D2N_args.maxit = 5;

%% quadrature parameters
qargs.n = 8;         % 2n points sampled per edge
qargs.kress_sig = 7;  % Kress quadrature parameter \sigma > 2

%% build quadrature
quad = BuildQuads(qargs);

%% build element K
elm = BuildElement(dargs,quad);

%% define u
x = elm.x(:,1);
y = elm.x(:,2);

% u_exact = 0.5*log(x.*x+y.*y);
% gx = x./(x.*x+y.*y);
% gy = y./(x.*x+y.*y);

u_exact = x./(x.*x+y.*y);
gx = (x.*x-y.*y)./(x.*x+y.*y).^2;
gy = -2*x.*y./(x.*x+y.*y).^2;

% u_exact = x;
% gx = ones(size(x));
% gy = zeros(size(x));

% u_exact = y;
% gx = zeros(size(x));
% gy = ones(size(y));

% u_exact = x-y;
% gx = ones(size(x));
% gy = -gx;

% u_exact = x.*y;
% gx = y;
% gy = x;

%% normal derivative
nx = elm.unit_normal(:,1);
ny = elm.unit_normal(:,2);
u_nd = gx.*nx + gy.*ny;

%% compute trace of u
u = SolveNeumannProblem(u_nd,elm,quad,D2N_args);

%% error
e = abs(u-u_exact);
maxe = max(e);
disp([qargs.n,maxe])

%% plots
figure(1),clf
    subplot(2,1,1)
        elm.plot_trace('plot-physical',u_exact,'ko-',quad)
        hold on
        elm.plot_trace('plot-physical',u,'r*-',quad)
        hold off
        grid minor
    subplot(2,1,2)
        elm.plot_trace('log-physical',e,'ko-',quad)
        title(sprintf('error: max = %.4e',maxe))
        grid minor
