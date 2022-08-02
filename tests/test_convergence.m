%%
clear
close all
format compact
format short e
addpath('classes','builders','PuncturedDomainHigherOrderQuadrature')

%% mesh cell parameters
dargs.id = 'square-circular-hole';
dargs.r_in = 0.25;
dargs.num_inner_edges = 2;

%% function space parameters
vargs.deg = 3; % polynomial degree, V_m(K)

%% Dirichlet-to-Neumann map parameters
D2N_args.type = 'density';
D2N_args.restart = 10;
D2N_args.tol = 1e-6;
D2N_args.maxit = 5;

%% quadrature parameters
qargs.n = [];         % 2n points sampled per edge
qargs.kress_sig = 7;  % Kress quadrature parameter \sigma > 2

%% pick v, w
% [v/v , v/e , e/e, b/b, v/b]
v = 1;
w = 1;
% v = [1,1,5,23,1];
% w = [2,5,7,23,23];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 1:numel(v)
    
    fprintf('\nk = %d\n\n',k);
    
    i = v(k);
    j = w(k);

    filename = 'experiments/square-circular-hole-%0.2d-%0.2d.txt';
    filename = sprintf(filename,i,j);
    fileID = fopen(filename,'w');

    if fileID < 0
        error('Cannot open file');
    end

    n_list = [8,16,32,64,128];
%     n_list = [8,16,32];
    for n = n_list

        fprintf('n = %d\n',n);

        qargs.n = n;

        %% build quadrature
        quad = BuildQuads(qargs);

        %% build element K
        elm = BuildElement(dargs,quad);

        %% generate a basis of V_m(K)
        vmk = class_local_function_space;
        vmk = vmk.init(elm,quad,vargs);

        %%
        f = vmk.f(:,i);
        p = vmk.p{i};

        g = vmk.f(:,j);
        q = vmk.p{j};

        z = [0,0];

        %% compute integrals
        [L2,H1] = PuncturedHighOrderQuadrature(elm,quad,z,p,f,q,g,D2N_args);

        %% print to terminal
        str = '&\t %4d\t &\\textbf{%.12e}\t &\\textbf{%.12e} \\\\ \n';
        fprintf(fileID,str,n,L2,H1);

    end

    fclose(fileID);

end

beep







