%%
clear
close all
format compact
format short e

%% mesh cell parameters
args.id = 'square';
% args.id = 'square-circular-hole';
args.r_in = 0.25;
args.num_inner_edges = 2;

%% parameters
args.n = 32;         % 2n points sampled per edge
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
args.restart = 10;
args.tol = 1e-12;
args.maxit = 10;

%%
x = elm.x(:,1);
y = elm.x(:,2);
un1 = elm.unit_normal(:,1);
un2 = elm.unit_normal(:,2);

% U_exact = x.^3 - 3*x.*y.^2;
% gx = 3*(x.^2-y.^2);
% gy = -6*x.*y;

U_exact = exp(x).*sin(y);
gx = U_exact;
gy = exp(x).*cos(y);

% center = [2,0];
% U_exact = 0.5*log((x-center(1)).^2+(y-center(2)).^2);
% gx = (x-center(1))./((x-center(1)).^2+(y-center(2)).^2);
% gy = (y-center(2))./((x-center(1)).^2+(y-center(2)).^2);


dUdn = gx.*un1+gy.*un2;
wgt = elm.get_dx_wgt(quad);
dUdn_wgt = dUdn.*wgt;

tic
U = GetTraceFromNeumannData(dUdn_wgt,elm,quad,args);
toc

U_error = abs(U-U_exact);

%% Compute H1 seminorm of u
H1 = sqrt(sum(U.*dUdn_wgt)*(pi/quad{1}.n));
H1_exact = sqrt(0.972651479839);
H1_error = abs(H1-H1_exact);
disp(H1_error)

%%
figure(2),clf
elm.plot_trace('plot-physical',U_exact,'bo-',quad);
hold on
elm.plot_trace('plot-physical',U,'ko-',quad);
hold off

figure(3),clf
elm.plot_trace('log-physical',U_error,'ko-',quad);

figure(4),clf
elm.plot_trace('log',U_error,'ko-',quad);


