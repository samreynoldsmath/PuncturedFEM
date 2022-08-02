function [A,M] = LocalStiffnessMassMatrices(...
    vmk,elm,quad,d2n_args,do_L2,do_H1)

    disp('Computing local stiffness and mass matrices...')

    %% include dependencies
    addpath('PuncturedDomainHigherOrderQuadrature');

    %% allocations
    A = zeros(vmk.dim,vmk.dim); % local stiffness matrix
    M = zeros(vmk.dim,vmk.dim); % local mass matrix
    
    %% to track progress
    num_comps = vmk.dim*(vmk.dim+1)/2;
    ctr = 0;
    
    %% message
    if do_L2 && do_H1
        msg = 'L2 and H1';
    elseif do_L2
        msg = 'L2';
    elseif do_H1
        msg = 'H1';
    else
        msg = 'nothing';
    end
    
    %% Local stiffness and mass matrices
    z = [0,0];
    for i = 1:vmk.dim
        
        % function v with boundary trace f and laplacian p
        f = vmk.f(:,i);
        p = vmk.p{i};
        
        for j = i:vmk.dim
            
            ctr = ctr+1;
            fprintf("Computing %s: %4d / %4d\n",msg,ctr,num_comps);
            
            % function w with boundary trace g and laplacian q
            g = vmk.f(:,j);
            q = vmk.p{j};
            
            % compute L2 inner product, H1 semi-inner product
            [L2,H1] = InnerProducts(f,p,g,q,z,elm,quad,d2n_args,do_L2,do_H1);
            
            % stiffness matrix
            A(i,j) = H1;
            A(j,i) = H1;
            
            % mass matrix 
            M(i,j) = L2;
            M(j,i) = L2;
        
        end
    end    
    
end