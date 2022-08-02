function [V,a] = HarmonicConjugateFromDoubleLayerDensity(...
    u,elm,quad,gmres_args)

    %{
        u: Dirichlet boundary trace of a harmonic function
        elm: object of type class_element
        quad: 2x1 cell array of type class_quad
            quad{1}: trapezoid rule
            quad{2}: Kress quadrature
        gmres_param: parameters for GMRES solver
            (restart, tol, maxit)
    %}
    
    % Right-hand side
    b = [2*u;zeros(elm.num_segs-1,1)];
    
    % Matrix
    K = Kfun(elm,quad);
    B = Bfun(elm);
    C = Cfun(elm,quad);

    N = elm.num_pts;
    M = elm.num_segs-1;
    z_idx = elm.nearest_vertex_idx;
    A = @(x) Afun(x,K,B,C,z_idx,N,M);
    
    % solve linear system
    x = gmres(A,b,gmres_args.restart,gmres_args.tol,gmres_args.maxit);

    % break solution into components
    mu = x(1:elm.num_pts);
    a = x(elm.num_pts+1:elm.num_pts+elm.num_segs-1);
    
    % get derivative of density
    dmu = FFT_derivative_boundary_trace(mu,elm);
    
    % Compute harmonic conjugate and its derivative
    V = HarmonicConjugateFromDensity(mu,dmu,elm,quad);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = Afun(x,K,B,C,z_idx,N,M)

    mu = x(1:N);
    muz = mu(z_idx);    % values of mu at the corners
    a = x(N+1:N+M);     % logarithmic coefficients
    y = zeros(N+M,1);   % allocate space for evaluation

    y(1:N) = (mu+muz) + K*mu - muz.*(K*ones(N,1)) + B*a;
    y(N+1:N+M) = C*mu;
     
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function B = Bfun(elm)

    B = zeros(elm.num_pts,elm.num_segs-1);
    
    for edgei = 1:elm.num_edges
        pt_i = elm.get_edge_point_idx(edgei);
        for i = pt_i
            for segj = 2:elm.num_segs
                XP = elm.x(i,:)-elm.seg_int_pt(segj,:);
                B(i,segj-1) = 2*log(norm(XP));
            end
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C = Cfun(elm,quad)
    
    C = zeros(elm.num_segs-1,elm.num_pts);
    
    for seg = 2:elm.num_segs
        edge_list = elm.seg_edge_idx{seg};
        for edge = edge_list
            edge_pt_idx = elm.get_edge_point_idx(edge);
            q = elm.edge_quad_type(edge);
            C(seg-1,edge_pt_idx) = elm.dx_norm(edge_pt_idx).*quad{q}.wgt;
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute trace of harmonic conjugate
function V = HarmonicConjugateFromDensity(mu,dmu,elm,quad)

    V = zeros(elm.num_pts,1);

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