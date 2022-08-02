classdef class_local_function_space
    
    properties 
        
        deg             % polynomial degree m
        dim             % dim V_m(K)
        num_vfun        % number of vertex functions
        num_efun        % number of edge functions
        num_pfun        % number of (cannonical) polynomial edge functions
        num_bfun        % number of bubble functions
        vfun_idx        % indices of vertex functions
        efun_idx        % indices of edge functions
        pfun_idx        % indices of polynomial functions
        bfun_idx        % indices of bubble functions
        type            % type index for the ith function
                            % 0: vertex function
                            % 1: edge function
                            % 2: bubble function
        pos             % vertex/edge associated with ith function
                            % for each segment:
                                % * vertices first, labeled CCW
                                % * edges next, labeled counter clockwise
                            % bubble functions last, labeled zero
        tag             % two functions share a tag if they are 
                            % form part of the same global basis function
        sis_fun         % sister functions share a tag
        f               % boundary trace values for each basis function
        p               % cell array of polynomial objects corresponding
                            % to the laplacians of each basis function 
    end
    
    methods 
        
        %% initilization
        function self = init(self,elm,quad,args)
            
            % polynomial degree
            self.deg = args.deg;
            
            % initialize function space dimension to max possible
            self = self.compute_max_dim(elm);
            
            % compute traces for harmonic functions
            self = self.get_boundary_traces(elm);
            
            % elliminate dependencies to get a basis
            self = self.reduce(elm,quad);
            
            % initialize all Laplacians to zero polynomial objects
            self.p = cell(1,self.dim);
            for k = 1:self.dim
                self.p{k} = class_poly;
                self.p{k} = self.p{k}.zero_poly();
            end
            
            % bubble functions: nontrivial laplacians, zero trace
            for k = 1:self.num_bfun
                idx = k-1;
                coef = -1;
                self.p{self.bfun_idx(k)} = ...
                    self.p{self.bfun_idx(k)}.init(idx,coef);
            end
            
            % sister functions
            self.sis_fun = self.get_sister_funs();
            
        end
                
        %% plot traces and print Laplacians
        function print(self,fig_num,elm,quad)
            for k = 1:self.dim
                figure(fig_num+k-1)
                elm.plot_trace('plot-physical',self.f(:,k),'ko-',quad)
                fprintf("\tk = %4d: \t",k);
                self.p{k}.print();
            end 
        end
        
        %% number of bubble functions
        function num_bfun = compute_number_bubbles(self)
            num_bfun = self.deg*(self.deg-1)/2;
        end
        
        %% maximum allowed dimension of the local function space
        function self = compute_max_dim(self,elm)
            m = self.deg;
%             self.dim = ( m*(m-1)+(m+2)*(m+1) )/2;
            num_closed_edges = elm.get_num_closed_edges();
            self.num_bfun = self.compute_number_bubbles();
            self.num_pfun = num_closed_edges*((m+2)*(m+1))/2;
            self.num_vfun = elm.num_edges;
            self.num_efun = elm.num_edges*((m+2)*(m+1))/2;
            self.dim = self.num_vfun ...
                + self.num_efun ...
                + self.num_pfun ...
                + self.num_bfun;
        end
        
        %%
        function idx = get_idx_type(self,t)
            idx = [];
            for k = 1:self.dim
                if self.type(k) == t
                    idx = [idx,k];
                end
            end
        end
        
        %%
        function self = get_all_type_idx(self)
            self.vfun_idx = self.get_idx_type(0);
            self.efun_idx = self.get_idx_type(1);
            self.pfun_idx = self.get_idx_type(-1);
            self.bfun_idx = self.get_idx_type(2);
            
            self.num_vfun = numel(self.vfun_idx);
            self.num_efun = numel(self.efun_idx);
            self.num_pfun = numel(self.pfun_idx);
            self.num_bfun = numel(self.bfun_idx);
        end
        
        %% canonical basis of P_m(e) for segments with single edge
        % 1, x, y, x^2, xy, y^2, etc.
        function F = edge_canonical_polys(self,elm,edge)
            pt_idx = elm.get_edge_point_idx(edge);
            m = self.deg;
            num_polys = ((m+2)*(m+1))/2;
            F = zeros(numel(pt_idx),num_polys);
            for k = 1:num_polys
                idx = k-1;
                coef = 1;
                P = class_poly;
                P = P.init(idx,coef);
                X = elm.x(pt_idx,1);
                Y = elm.x(pt_idx,2);
                Z = [0,0];
                F(:,k) = P.eval(X,Y,Z);
            end
        end
        
        %%
        function ell = edge_barycentric_linear(self,elm,edge)
            
            % edge indices and sizes
            edge_idx = elm.get_edge_point_idx(edge);
            num_pts = numel(edge_idx);
            sqrt3 = sqrt(3);
            
            % allocations
            ell = zeros(num_pts,3);
            z = zeros(2,3); 
            
            % rotation
            R = [0,1;-1,0];
            
            % vertices
            z(:,1) = elm.x(elm.edge_idx(edge),:);
            z(:,2) = elm.get_next_vertex(edge);
            z(:,3) = 0.5*(z(:,1)+z(:,2)-sqrt3*R*(z(:,2)-z(:,1)));
            
            % barycentric coordinates
            h = norm(z(:,2)-z(:,1));
            for j = 1:3
                jm = mod(j-2,3)+1; % j-1 mod 3
                jp = mod(j,3)+1; % j+1 mod 3
                xz = elm.x(edge_idx,:);
                xz(:,1) = xz(:,1)-z(1,j);
                xz(:,2) = xz(:,2)-z(2,j);
                rzz = 2*R*(z(:,jm)-z(:,jp))/(sqrt3*h*h);
                for k = 1:numel(edge_idx)
                    ell(k,j) = 1-dot(xz(k,:),rzz);
                end
            end
        end
        
        %% 
        function [vfun_traces,vfun_pos,efun_traces,efun_pos,efun_tag] ...
                = linear_traces(self,elm)
            
            num_pts = elm.num_pts;
            vfun_traces = zeros(num_pts,elm.num_edges);
            efun_traces = zeros(num_pts,elm.num_edges);
            
            vfun_pos = zeros(1,elm.num_edges);
            efun_pos = zeros(1,elm.num_edges);
            
            efun_tag = zeros(1,elm.num_edges);
            
            sis_edges = elm.get_all_sister_edges();
            
            for seg = 1:elm.num_segs
                % only find vertex and edge functions on segments with 
                % more than one edge
                num_edges = numel(elm.seg_edge_idx{seg});
                if num_edges > 1
                    
                    e1 = elm.seg_edge_idx{seg}(1);
                    for edge = elm.seg_edge_idx{seg}
                        
                        % indices of points on this edge
                        pt_idx = elm.get_edge_point_idx(edge);
                        
                        % loop edges on a segment back around
                        edge_plus1 = mod(edge+1-e1,num_edges)+e1;
                        
                        % get barycentric coordinates on this edge
                        ell = self.edge_barycentric_linear(elm,edge);
                        
                        % glue vertex function traces together
                        vfun_traces(pt_idx,edge) = ell(:,1);
                        vfun_traces(pt_idx,edge_plus1) = ell(:,2);
                        
                        % (linear) edge functions
                        sis_edge = sis_edges{edge}(1);
                        if sis_edge > edge
                            % new edge type
                            efun_traces(pt_idx,edge) = ell(:,3);
                        else
                            % already seen this edge
                            sis_pt_idx = elm.get_edge_point_idx(sis_edge);
                            efun_traces(pt_idx,edge) ...
                                = flip(efun_traces(sis_pt_idx,sis_edge));
                        end
                        efun_tag(edge) = elm.edge_tag(edge);
                        
                        % set position labels
                        vfun_pos(edge) = edge;
                        efun_pos(edge) = edge;
                        
                    end
                end
            end
        end
        
        %%
        function [qfun_traces,qfun_pos,qfun_tag] = quadratic_traces(self,elm)
            
            num_pts = elm.num_pts;
            qfun_traces = zeros(num_pts,3*elm.num_edges);
            
            qfun_pos = zeros(1,3*elm.num_edges);
            qfun_tag = zeros(1,3*elm.num_edges);
            sis_edges = elm.get_all_sister_edges();
            T = max(elm.edge_tag);
            
            for seg = 1:elm.num_segs
                % only find vertex and edge functions on segments with 
                % more than one edge
                num_edges = numel(elm.seg_edge_idx{seg});
                if num_edges > 1
                    
                    for edge = elm.seg_edge_idx{seg}
                        
                        % indices of points on this edge
                        pt_idx = elm.get_edge_point_idx(edge);
                        
                        sis_edge = sis_edges{edge}(1);
                        if sis_edge > edge
                            % new edge type
                            
                            % get barycentric coordinates on this edge
                            ell = self.edge_barycentric_linear(elm,edge);
                            
                            % products of linear functions
                            qfun_traces(pt_idx,3*(edge-1)+1) ...
                                = 4*ell(:,1).*ell(:,2);
                            qfun_traces(pt_idx,3*(edge-1)+2) ...
                                = 4*ell(:,1).*ell(:,3);
                            qfun_traces(pt_idx,3*(edge-1)+3) ...
                                = 4*ell(:,2).*ell(:,3);
                            
                        else
                            % already seen this edge
                            sis_pt_idx = elm.get_edge_point_idx(sis_edge);
                            for j = 1:3
                                k = 3*(edge-1)+j;
                                ks = 3*(sis_edge-1)+j;
                                qfun_traces(pt_idx,k) ...
                                    = flip(qfun_traces(sis_pt_idx,ks));
                            end
                        end
                        
                        % position
                        for j = 1:3
                            k = 3*(edge-1)+j;
                            qfun_pos(k) = edge;
                        end
                        
                        % tags
                        for j = 1:3
                            k = 3*(edge-1)+j;
                            qfun_tag(k) = j*T+elm.edge_tag(edge);
                        end
                        
                        
                    end
                end
            end
        end
        
        %%
        function [cfun_traces,cfun_pos,cfun_tag] = cubic_traces(self,elm)
            
            num_pts = elm.num_pts;
            cfun_traces = zeros(num_pts,4*elm.num_edges);
            
            cfun_pos = zeros(1,4*elm.num_edges);
            cfun_tag = zeros(1,4*elm.num_edges);
            sis_edges = elm.get_all_sister_edges();
            T = max(elm.edge_tag);
            
            for seg = 1:elm.num_segs
                % only find vertex and edge functions on segments with 
                % more than one edge
                num_edges = numel(elm.seg_edge_idx{seg});
                if num_edges > 1
                    
                    for edge = elm.seg_edge_idx{seg}
                        
                        % indices of points on this edge
                        pt_idx = elm.get_edge_point_idx(edge);
                        
                        sis_edge = sis_edges{edge}(1);
                        if sis_edge > edge
                            % new edge type
                            
                            % get barycentric coordinates on this edge
                            ell = self.edge_barycentric_linear(elm,edge);
                            
                            cfun_traces(pt_idx,4*(edge-1)+1) ...
                                = (1.5*sqrt(3))*ell(:,1).*ell(:,2)...
                                    .*(ell(:,1)-ell(:,2));
                            cfun_traces(pt_idx,4*(edge-1)+2) ...
                                = (1.5*sqrt(3))*ell(:,1).*ell(:,3)...
                                    .*(ell(:,1)-ell(:,3));
                            cfun_traces(pt_idx,4*(edge-1)+3) ...
                                = (1.5*sqrt(3))*ell(:,2).*ell(:,3)...
                                    .*(ell(:,2)-ell(:,3));
                            cfun_traces(pt_idx,4*(edge-1)+4) ...
                                = 27*ell(:,1).*ell(:,2).*ell(:,3);
                            
                        else
                            % already seen this edge
                            sis_pt_idx = elm.get_edge_point_idx(sis_edge);
                            for j = 1:3
                                k = 4*(edge-1)+j;
                                ks = 4*(sis_edge-1)+j;
                                cfun_traces(pt_idx,k) ...
                                    = flip(cfun_traces(sis_pt_idx,ks));
                            end
                        end
                        
                        % position
                        for j = 1:4
                            k = 4*(edge-1)+j;
                            cfun_pos(k) = edge;
                        end
                        
                        % tags
                        for j = 1:4
                            k = 4*(edge-1)+j;
                            cfun_tag(k) = (j+3)*T+elm.edge_tag(edge);
                        end
                        
                    end
                end
            end
        end
        
        %% 
        function [pfun_traces,closed_edges,pfun_tag] = ...
                polynomial_traces(self,elm)
            pfun_traces = zeros(elm.num_pts,self.num_pfun);
            closed_edges = elm.get_closed_edge_idx();
            pfun_tag = zeros(1,self.num_pfun);
            T = max(elm.edge_tag);
            k = 0;
            m = self.deg;
            local_dim = ((m+2)*(m+1))/2;
            for edge = closed_edges
                k = k(end)+( 1:local_dim );
                pt_idx = elm.get_edge_point_idx(edge);
                pfun_traces(pt_idx,k) ...
                    = self.edge_canonical_polys(elm,edge);
                pfun_tag(k) = -k*T+elm.edge_tag(edge);
            end
            
        end
        
        %%
        function self = get_boundary_traces(self,elm)
            
            self.f = zeros(elm.num_pts,self.dim);
            self.type = zeros(1,self.dim);
            self.pos = zeros(1,self.dim);
            self.tag = zeros(1,self.dim);
            
            % define indices for functions
            self.vfun_idx = 1:elm.num_edges;
            self.pfun_idx = self.vfun_idx(end) + ( 1:self.num_pfun );
            self.efun_idx = self.pfun_idx(end) + ( 1:self.num_efun );
            self.bfun_idx = self.efun_idx(end) + ( 1:self.num_bfun );
            
            % linear functions
            [vfun_traces,vfun_pos,efun_traces,efun_pos,efun_tag] ...
                = self.linear_traces(elm);
            
            % quadratic functions
            if self.deg > 1
                [qfun_traces,qfun_pos,qfun_tag] = ...
                    self.quadratic_traces(elm);
            else
                qfun_traces = [];
                qfun_pos = [];
                qfun_tag = [];
            end
            
            % cubic functions
            if self.deg > 2
                [cfun_traces,cfun_pos,cfun_tag] = self.cubic_traces(elm);
            else
                cfun_traces = [];
                cfun_pos = [];
                cfun_tag = [];
            end
            
            % polynomial functions
            [pfun_traces,pfun_pos,pfun_tag] = self.polynomial_traces(elm);
            
            % assign all traces
            self.f(:,self.vfun_idx) = vfun_traces;
            m = size([efun_traces,qfun_traces,cfun_traces],2);
            self.f(:,self.efun_idx(1:m)) = [...
                efun_traces,...
                qfun_traces,...
                cfun_traces ];
            self.f(:,self.pfun_idx) = pfun_traces;
            
            % assign type labels
            self.type(self.vfun_idx) = 0;
            self.type(self.efun_idx) = 1;
            self.type(self.pfun_idx) = -1;
            self.type(self.bfun_idx) = 2;
            
            % assign position labels
            self.pos(self.vfun_idx) = vfun_pos;
            self.pos(self.efun_idx(1:m)) = [efun_pos,qfun_pos,cfun_pos];
            self.pos(self.pfun_idx) = pfun_pos;
            
            % assign tags for harmonic functions
            self.tag(self.vfun_idx) = 0;
            self.tag(self.efun_idx(1:m)) = [efun_tag,qfun_tag,cfun_tag];
            self.tag(self.pfun_idx) = pfun_tag;
            
            % assign tags for bubble functions
            T = max(self.tag);
            for k = self.bfun_idx
                self.tag(k) = k+T;
            end
            
        end
                
        %% reduce spanning set of harmonic functions to a basis
        function self = reduce(self,elm,quad)
            
            % form Gram matrix
            M = zeros(self.dim,self.dim);
            for i = 1:self.dim
                for j = 1:self.dim
                    fg = self.f(:,i).*self.f(:,j);
                    M(i,j) = elm.integrate_over_boundary(fg,quad);
                    M(j,i) = M(i,j);
                end
            end
            
            % get basis from spanning set
            tol = 1e-12;
            basis_idx = self.get_basis_idx(M,tol);
            
            % keep bubbles
            basis_idx = [basis_idx,self.bfun_idx];
            
            % reassign basis functions 
            self.f = self.f(:,basis_idx);
            self.type = self.type(basis_idx);
            self.pos = self.pos(basis_idx);
            self.tag = self.tag(basis_idx);
            
            % get new dimensions
            self = self.get_dimensions();
            
            % label each type of function (vertex, edge, bubble)
            self = self.get_all_type_idx();
            
        end
        
        %% Given a Gram matrix of a spanning set, reduce to a basis
        function idx = get_basis_idx(self,M,tol)
            idx = [];
            [mkk,k] = max(diag(M));
            while mkk > tol
                idx = [idx,k];
                M = M - M(:,k)*M(k,:)/mkk;
                [mkk,k] = max(diag(M));
            end
            idx = sort(idx);
        end
        
        %%
        function self = get_dimensions(self)
            self.dim = size(self.f,2);
            self.num_efun = self.dim - self.num_bfun - self.num_vfun;
        end
        
        %%
        function sis = get_sister_funs(self)
            sis = cell(1,self.dim);
            for k = 1:self.dim
                t = self.tag(k);
                sis{k} = find(t == self.tag);
                sis{k}(sis{k} == k) = [];
            end
        end
        
        
    end
    
end