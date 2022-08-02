function u = SolveNeumannProblem(u_nd,elm,quad,gmres_args)

    %{
        Adapted from Kress's "Linear Integral Equations", Thm 6.26.

        u_nd: normal derivative of a harmonic function
        elm: object of type class_element
        quad: 2x1 cell array of type class_quad
            quad{1}: trapezoid rule
            quad{2}: Kress quadrature
        gmres_param: parameters for GMRES solver
            (restart, tol, maxit)
    %}

    u_exists = check_existence_nuemann_problem(u_nd,elm,quad);
    
    if ~u_exists
        warning('Solution does not exist for Neumann problem')
    end
    
    % Right-hand side
    b = [2*u_nd;zeros(elm.num_segs-1,1)];
    
    % Matrix
    K = Kfun(elm,quad);
    B = Bfun(elm);
    C = Cfun(elm,quad);

    N = elm.num_pts;
    M = elm.num_segs-1;
    z_idx = elm.nearest_vertex_idx;
    A = @(x) Afun(x,K,B,C,z_idx,N,M);
    
    Amat = [eye(N)+K,B;C,zeros(M)];
    
    % solve linear system
    x = gmres(A,b,gmres_args.restart,gmres_args.tol,gmres_args.maxit);

    % break solution into components
    psi = x(1:elm.num_pts);
    a = x(elm.num_pts+1:elm.num_pts+elm.num_segs-1)
    
    % Compute harmonic conjugate and its derivative
    u = GetTraceFromDensity(psi,elm,quad);
    
    % Add logarithmic terms
    w = LogarithmicTraces(elm,a);
    u = u+w;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = Afun(x,K,B,C,z_idx,N,M)

    psi = x(1:N);
    psi_z = psi(z_idx);    % values of mu at the corners
    a = x(N+1:N+M);     % logarithmic coefficients
    y = zeros(N+M,1);   % allocate space for evaluation

    y(1:N) = psi + K*psi + B*a;
%     y(1:N) = (mu+muz) + K*mu - muz.*(K*ones(N,1)) + B*a;
    y(N+1:N+M) = C*psi;
     
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
                    dx = elm.dx(i,:);
                    ndx = elm.dx_norm(i);
                    ndy = elm.dx_norm(j);
                    if xy2 < 1e-12
                        ddy = elm.ddx(j,:);
                        K(i,j) = -0.5*( ddy(1)*dx(2) - ddy(2)*dx(1) )/ndx^2;
                    else
                        K(i,j) = (ndy/ndx)*( xy(1)*dx(2) - xy(2)*dx(1) )/xy2;   
                    end
                    
                    % quadrature weights
                    K(i,j) = (-1/pi)*(K(i,j)+ndy)*quad{q}.wgt(jj);
%                     K(i,j) = (-1/pi)*(K(i,j))*quad{q}.wgt(jj);
                    
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
function u = GetTraceFromDensity(psi,elm,quad)

    mat = zeros(elm.num_pts);

    u = zeros(elm.num_pts,1);

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
            npi = quad{q}.n/pi;
            
            if edgei == edgej
                %% Do Kress and Martensen

                % loop over points on edge i
                for i = pt_i

                    x = elm.x(i,:);

                    % loop over quadrature points on edge j
                    for j = pt_j

                        jj = j-elm.edge_idx(edgej)+1;
                        ij = abs(i-j)+1;

                        y = elm.x(j,:);
                        xy = x-y;
                        xy2 = dot(xy,xy);

                        if xy2 < tol
                            L1 = (0.5/pi)*log( ...
                                npi*quad{q}.wgt(jj)*elm.dx_norm(j) );
                        else
                            L1 = (0.25/pi)*log(xy2/quad{3}.t(ij));
                        end

                        dx_wgt = elm.dx_norm(j)*quad{q}.wgt(jj);
                        L2 = npi*quad{3}.wgt(ij);
                        u(i) = u(i) + psi(j)*(L1+L2)*dx_wgt;
                        
                        mat(i,j) = (L1+L2)*dx_wgt;

                    end
                end
                
            else % if elsei != elsej
                %% Do Kress only
                
                % loop over points on edge i
                for i = pt_i

                    x = elm.x(i,:);

                    % loop over quadrature points on edge j
                    for j = pt_j

                        jj = j-elm.edge_idx(edgej)+1;

                        y = elm.x(j,:);
                        xy = x-y;
                        xy2 = dot(xy,xy);

                        L = -(0.25/pi)*log(xy2);

                        dx_wgt = elm.dx_norm(j)*quad{q}.wgt(jj);
                        u(i) = u(i) + L*dx_wgt*psi(j);
                        
                        mat(i,j) = L*dx_wgt;

                    end
                end
                
            end
        end    
    end
    
    cond(mat)
    figure(39),imagesc(mat)
    
    x=elm.x(:,1);y=elm.x(:,2);u_exact=x;
    phi=mat\u_exact;
    figure(40),plot(phi,'ko-'),hold on,plot(psi,'r*-'),hold off
end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Right-hand side for the Nystrom system
function B=NystromRHS(u_nd,ELEMENT,MART,QUAD)    
    %{
        Computes right-hand side of the Nystrom system.
    %}

    % Tolerance for square norm
    TOL=1.0e-14;

    % For readability
    m=ELEMENT.num_edges;
    n=QUAD.n;
    
    % Allocate space for output
    B=zeros(2*m*n,1);
    
    % Precompute scaling, common product
    scale = -0.25/pi;
    npi=n/pi;
    
    % outer loop over edges
    for edgei=1:m
        % loop over points on edge i
        for i=1:2*n
            ii=i+(edgei-1)*(2*n);
            % inner loop over edges
            for edgej=1:m
                
                if edgei==edgej
                    
                    % Kress and Martensen quadratures
                    for j=2:2*n
                        
                        jj=j+(edgej-1)*(2*n);
                        ij=abs(i-j)+1;

                        xx=ELEMENT.x(ii,1)-ELEMENT.x(jj,1);
                        yy=ELEMENT.x(ii,2)-ELEMENT.x(jj,2);
                        xy2=xx*xx+yy*yy;

                        if xy2<TOL
                            L1=2*scale*log( npi*QUAD.wgt(j)...
                                *ELEMENT.dx_norm(jj) );
                        else
                            L1=scale*log( xy2/MART.t(ij) );
                        end

                        B(ii)=B(ii)+( L1+npi*MART.wgt(ij) )...
                            *ELEMENT.dx_norm(jj)*QUAD.wgt(j)*u_nd(jj);

                    end
                    
                else
                    
                    % Kress quadrature only
                    for j=2:2*n
                        
                        jj=j+(edgej-1)*(2*n);

                        xx=ELEMENT.x(ii,1)-ELEMENT.x(jj,1);
                        yy=ELEMENT.x(ii,2)-ELEMENT.x(jj,2);
                        xy2=xx*xx+yy*yy;

                        B(ii)=B(ii)+scale*log(xy2)...
                            *ELEMENT.dx_norm(jj)*QUAD.wgt(j)*u_nd(jj);

                    end
                    
                end
            end
        end    
    end
    
end
