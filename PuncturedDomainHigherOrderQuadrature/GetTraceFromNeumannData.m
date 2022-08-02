function [U,V] = GetTraceFromNeumannData(dUdn_wgt,elm,quad,D2N_args)
    %{
        Assumes U has a harmonic conjugate V
    %}

    % recover harmonic conjugate
    V = FFT_antiderivative(dUdn_wgt,elm);
    
    % solve for density
    Afun = @(x) HarmonicConjugateFromDensity(x,elm,quad);
    
    restart = D2N_args.restart;
    tol = D2N_args.tol;
    maxit = D2N_args.maxit;
    mu = gmres(Afun,V,restart,tol,maxit);
    
    % get trace
    N = elm.num_pts;
    z_idx = elm.nearest_vertex_idx;
    muz = mu(z_idx);
    K = Kfun(elm,quad);
    U = 0.5*((mu+muz) + K*mu - muz.*(K*ones(N,1)));
%     U = 0.5*(mu+K*mu);
end

%% Compute trace of harmonic conjugate
function V = HarmonicConjugateFromDensity(mu,elm,quad)

    V = zeros(elm.num_pts,1);
    
    dmu = FFT_derivative_boundary_trace(mu,elm);

    % Tolerance for square norm
    tol=1.0e-12;
    
    % loop over edges
    for edgei = 1:elm.num_edges
        
        % get indices for edge i
        pt_i = elm.get_edge_point_idx(edgei);
        
        for edgej = 1:elm.num_edges
            
            % get indices for edge j
            pt_j = elm.get_edge_point_idx(edgej);
            q = elm.edge_quad_type(edgej);
            
            % loop over points on edge i
            for i = pt_i
                
                x = elm.x(i,:);

                % loop over quadrature points on edge j
                for j = pt_j
                    
                    jj = j-elm.edge_idx(edgej)+1;
                    
                    y = elm.x(j,:);
                    xy = x-y;
                    xy2 = dot(xy,xy);

                    if xy2 < tol
                        L = -(0.5/quad{q}.n)*dmu(j);
                    else
                        dy = elm.dx(j,:)*quad{q}.wgt(jj);
                        L = (0.5/pi)*dot(xy,dy)*(mu(j)-mu(i))/xy2;
                    end
                    
                    V(i) = V(i) + L;

                end
            end
        end    
    end
 
end

%% 
function K = Kfun(elm,quad)

    K = zeros(elm.num_pts);
    
    for edgei = 1:elm.num_edges
        
        % get indices for edge i
        pt_i = elm.get_edge_point_idx(edgei);
        
        for edgej = 1:elm.num_edges
            
            % get indices for edge j
            pt_j = elm.get_edge_point_idx(edgej);
            q = elm.edge_quad_type(edgej);
            
            % loop over points on edge i
            for i = pt_i
                % loop over quadrature points on edge j
                for j = pt_j
                    jj = j-elm.edge_idx(edgej)+1;
                    xy = elm.x(i,:) - elm.x(j,:);
                    xy2 = dot(xy,xy);
                    dy = elm.dx(j,:);
                    ndy = elm.dx_norm(j);
                    if xy2 < 1e-12
                        ddy = elm.ddx(j,:);
                        K(i,j) = 0.5*( ddy(1)*dy(2) - ddy(2)*dy(1) )/ndy^2;
                    else
                        K(i,j) = ( xy(1)*dy(2) - xy(2)*dy(1) )/xy2;   
                    end
                    
                    % quadrature weights
                    K(i,j) = (-1/pi)*K(i,j)*quad{q}.wgt(jj);
                    
                end
            end
            
        end
        
    end
    
end