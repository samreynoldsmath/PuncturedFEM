%%
clear
close all
format compact
format short g

%% include dependencies
addpath('classes','builders','PuncturedDomainHigherOrderQuadrature')

%% mesh cell parameters
args.id = 'circle';
args.r = 1;
args.num_edges = 1;

%% parameters
args.n = 16;         % 2n points sampled per edge
args.kress_sig = 7;  % Kress quadrature parameter \sigma > 2

%% quadrature parameters
qargs.n = 32;         % 2n points sampled per edge
qargs.kress_sig = 7;  % Kress quadrature parameter \sigma > 2

%% build quadrature
quad = BuildQuads(qargs);

%% build element K
elm = BuildElement(args,quad);

disp(elm)
figure(1),clf,elm.plot_boundary('quiver')

%%
c = elm.x(:,1);
s = elm.x(:,2);
t = quad{1}.t;

%% fails for integrating constants
% dy = ones(size(c));
% y_exact = t;

%% gets exact value for sinusoids
% dy = 2*s.*c;
% y_exact = -0.5*(c.*c-s.*s);

%% does reasonable well with periodic polynomials
dy = 3*t.*t-6*pi*t+2*pi*pi;
y_exact = t.*(pi-t).*(2*pi-t);

y = FFT_antiderivative(dy,elm);

figure(2),clf
elm.plot_trace('plot',dy,'ko-',quad);
title('dy')

figure(3),clf
elm.plot_trace('plot',y_exact,'bo-',quad);
hold on
elm.plot_trace('plot',y,'ko-',quad);
hold off
title('y')

figure(4),clf
elm.plot_trace('log',abs(y-y_exact),'ko-',quad);
title('error')