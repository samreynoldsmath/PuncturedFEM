function [a,m] = LocalRHS(vmk,elm,quad,Q,D2N_args)

    disp('Computing local right-hand side vector...')

    %% include dependencies
    addpath('PuncturedDomainHigherOrderQuadrature');

    %% allocations
    a = zeros(vmk.dim,1); % local stiffness matrix
    m = zeros(vmk.dim,1); % local mass matrix
    
    %% to track progress
    ctr = 0;
    
    %% Local stiffness and mass matrices
    z = [0,0];
    for i = 1:vmk.dim
            
        ctr = ctr+1;
        fprintf("Computing L2 and H1: %4d / %4d\n",ctr,vmk.dim);
        
        % function w with boundary trace g and laplacian q
        f = vmk.f(:,i);
        p = vmk.p{i};
            
        % compute L2 inner product, H1 semi-inner product
        [L2,H1] = InnerProductWithPolynomial(f,p,Q,z,elm,quad,D2N_args);
        
        a(i) = H1;
        m(i) = L2;
        
    end    
end