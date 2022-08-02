%{
    class_element
    
    comments...

%}

classdef class_element
    properties
        id              % string or integer label
        x               % physical boundary points
        dx              % derivatives
        ddx             % second derivatives
        dx_norm         % norm of derivative
        unit_tangent    % unit tangent vector (parallel to dx)
        unit_normal     % outward unit normal vector
        arc_length      % total boundary length
        num_pts         % total number of sampled points
        num_edges       % number of edges
        edge_idx        % starting index for each edge
        edge_quad_type  % 1 for trapezoid rule, 2 for Kress quadrature
        num_segs        % number of segments
        seg_idx         % starting index for each segment
        seg_int_pt      % point lying in the interior of a hole
        seg_edge_idx    % edge labels for each segment
        edge_tag        % tag edges which will be joined in global mesh
    end
    methods
        
        %%
        function self = init(self,args,quad)
            
            % identifier and quadrature type labels
            self.id = args.id;
            self.edge_quad_type = (args.edge_quad_type(:))';
            
            % number of edges
            self.num_edges = numel(args.x_fun);
            
            % total number of points sampled on the boundary
            self.num_pts = 0;
            for q = 1:numel(quad)
                self.num_pts = self.num_pts ...
                    + 2*sum(self.edge_quad_type == q)*quad{q}.n;
            end
            
            % checks
            % [will perform size checks here...]
            
            % allocations
            self.x = zeros(self.num_pts,2);
            self.dx = zeros(self.num_pts,2);
            self.ddx = zeros(self.num_pts,2);
            self.unit_tangent = zeros(self.num_pts,2);
            self.unit_normal = zeros(self.num_pts,2);
            self.edge_idx = zeros(1,self.num_edges+1);
            
            % find starting points of edges
            self.edge_idx(1) = 1;
            for edge = 2:self.num_edges
                q = self.edge_quad_type(edge-1);
                self.edge_idx(edge) = self.edge_idx(edge-1) + 2*quad{q}.n;
            end
            self.edge_idx(self.num_edges+1) = self.num_pts+1;
            
            % find starting points of segments
            tol = 1e-6;
            self.num_segs = 0;
            self.seg_idx = [];
            i = 1;
            while i <=self.num_edges
                a = args.x_fun{i}(0);
                j = i;
                while j <= self.num_edges
                    b = args.x_fun{j}(2*pi);
                    d = norm(a-b);
                    % check if starting and ending points are the same
                    if d < tol
                        self.num_segs = self.num_segs + 1;
                        self.seg_idx = [self.seg_idx, self.edge_idx(i)];
                        i = j;
                        j = self.num_edges+1;
                    else
                        j = j+1;
                    end
                end
                i = i+1;
            end
            self.seg_idx = [self.seg_idx, self.num_pts+1];
            
            % find edge labels for each segment
            self.seg_edge_idx = cell(1,self.num_segs);
            for seg = 1:self.num_segs
                seg_pt_idx = self.get_seg_point_idx(seg);
                self.seg_edge_idx{seg} = [];
                for edge = 1:self.num_edges
                    edge_pt_idx = self.get_edge_point_idx(edge);
                    if seg_pt_idx(1) <= edge_pt_idx(1) ...
                            && edge_pt_idx(1) < seg_pt_idx(end)
                        self.seg_edge_idx{seg} = ...
                            [self.seg_edge_idx{seg},edge];
                    end
                end
            end
            
            % points lying in the interior of a hole
            if numel(args.seg_int_pt) ~= 2*(self.num_segs-1)
                error('Incorrect number of interior points of holes')
            end
            self.seg_int_pt = zeros(self.num_segs,2);
            self.seg_int_pt(2:self.num_segs,:) = args.seg_int_pt;
            
            
            % get boundary points and derivatives
            for edge = 1:self.num_edges
                
                % pick appropriate parameter for quadrature
                q = self.edge_quad_type(edge);
                t = quad{q}.t;
                
                % get indices corresponding to point on edge
                pt_idx = self.get_edge_point_idx(edge);
                                
                % get parameterization of boundary and derivatives
                self.x(pt_idx,:) = args.x_fun{edge}(t);
                self.dx(pt_idx,:) = args.dx_fun{edge}(t);
                self.ddx(pt_idx,:) = args.ddx_fun{edge}(t);
                
            end
            
            % compute unit tangent and unit normal vectors
            self.dx_norm = sqrt(self.dx(:,1).^2+self.dx(:,2).^2);
            self.unit_tangent(:,1) = self.dx(:,1)./self.dx_norm;
            self.unit_tangent(:,2) = self.dx(:,2)./self.dx_norm;
            self.unit_normal(:,1) = self.unit_tangent(:,2);
            self.unit_normal(:,2) = -self.unit_tangent(:,1);

            % get total arc length
            if isfield(args,'arc_length')
                self.arc_length = args.arc_length;
            else
                self = self.compute_arc_length(quad);
            end
            
            % set tags (for paired edges in global mesh)
            self.edge_tag = args.edge_tag;
        end
        
        %%
        function pt_idx = get_edge_point_idx(self,edge)
            pt_idx = self.edge_idx(edge):self.edge_idx(edge+1)-1;
        end
        
        %%
        function pt_idx = get_seg_point_idx(self,seg)
            pt_idx = self.seg_idx(seg):self.seg_idx(seg+1)-1;
        end
        
        %%
        function self = compute_arc_length(self,quad)
            self.arc_length = 0;
            for edge = 1:self.num_edges
                pt_idx = self.get_edge_point_idx(edge);
                q = self.edge_quad_type(edge);
                self.arc_length = self.arc_length ...
                    + sum(self.dx_norm(pt_idx).*quad{q}.wgt);
            end
        end
        
        %%
        function z_idx = nearest_vertex_idx(self)
            z_idx = zeros(self.num_pts,1);
            for seg = 1:self.num_segs
                seg_pt_idx = self.get_seg_point_idx(seg);
                for edge = self.seg_edge_idx{seg}(1:end-1)
                    edge_pt_idx = self.get_edge_point_idx(edge);
                    n = ceil(numel(edge_pt_idx)/2);
                    z_idx(edge_pt_idx(1:n)) = edge_pt_idx(1);
                    z_idx(edge_pt_idx(n+1:end)) = edge_pt_idx(end)+1;
                end
                edge = self.seg_edge_idx{seg}(end);
                edge_pt_idx = self.get_edge_point_idx(edge);
                n = ceil(numel(edge_pt_idx)/2);
                z_idx(edge_pt_idx(1:n)) = edge_pt_idx(1);
                z_idx(edge_pt_idx(n+1:end)) = seg_pt_idx(1);
            end
        end
        
        %% 
        function seg = get_seg_from_edge(self,edge)
            for s = 1:self.num_segs
                seg_edges = self.seg_edge_idx{s};
                for e = seg_edges
                    if e == edge
                        seg = s;
                    end
                end
            end
        end
        
        %%
        function z_idx = get_next_vertex_idx(self,edge)
            
            % move to the next edge on this segment
            seg = self.get_seg_from_edge(edge);
            edges_on_this_seg = self.seg_edge_idx{seg};
            first_edge = edges_on_this_seg(1);
            num_edges_on_this_seg = numel(edges_on_this_seg);
            k = mod(edge-first_edge+1,num_edges_on_this_seg)+first_edge;
            z_idx = self.edge_idx(k);
        end
        
        %%
        function z = get_next_vertex(self,edge)
            
            z_idx = self.get_next_vertex_idx(edge);
            z = self.x(z_idx,:);
            
        end
        
        %% 
        function seg = seg_of_edge(self,edge)
            for s = 1:self.num_segs
                for e = self.seg_edge_idx{s}
                    if e == edge
                        seg = s;
                    end
                end
            end
        end
        
        %%
        function num_closed_edges = get_num_closed_edges(self)
            closed_edges = self.get_closed_edge_idx();
            num_closed_edges = numel(closed_edges);
        end
        
        %% 
        function closed_edges = get_closed_edge_idx(self)
            closed_edges = [];
            for seg = 1:self.num_segs
                idx = self.seg_edge_idx{seg};
                if numel(idx) == 1
                    closed_edges = [closed_edges,idx(1)];
                end
            end
        end
        
        %%
        function same_seg = edges_on_same_seg(self,edge1,edge2)
            seg1 = self.seg_of_edge(edge1);
            seg2 = self.seg_of_edge(edge2);
            same_seg = ( seg1 == seg2);
        end
        
        %%
        function sis_edges = get_sister_edges(self,edge)
            sis_edges = find(self.edge_tag == self.edge_tag(edge));
            sis_edges(sis_edges == edge) = [];
        end
        
        %%
        function sis_edges = get_all_sister_edges(self)
            sis_edges = cell(1,self.num_edges);
            for edge = 1:self.num_edges
                sis_edges{edge} = self.get_sister_edges(edge);
            end
        end
        
        %%
        function f_trace = get_trace_values(self,f_fun)
            f_trace = zeros(self.num_pts,1);
            for i = 1:self.num_pts
                f_trace(i) = f_fun(self.x(i,1),self.x(i,2));
            end
        end
        
        %%
        function val = integrate_over_edge(self,f,edge,wgt)
            idx = self.get_edge_point_idx(edge);
            val = sum(wgt(:).*self.dx_norm(idx).*f(:));
        end
        
        %% 
        function val = integrate_over_seg(self,f,seg,quad)
            val = 0;
            for edge = self.seg_edge_idx{seg}
                fidx = self.get_edge_point_idx(edge);
                fidx = fidx+1-self.seg_idx(seg);
                wgt = quad{self.edge_quad_type(edge)}.wgt;
                val = val + self.integrate_over_edge(f(fidx),edge,wgt);
            end
        end
        
        %%
        function val = integrate_over_boundary(self,f,quad)
            val = 0;
            for seg = 1:self.num_segs
                fidx = self.get_seg_point_idx(seg);
                val = val + self.integrate_over_seg(f(fidx),seg,quad);
            end
        end
        
        %%
        function dx_wgt = get_dx_wgt(self,quad)
            dx_wgt = zeros(self.num_pts,1);
            for seg = 1:self.num_segs
                for edge = self.seg_edge_idx{seg}
                    edge_pt_idx = self.get_edge_point_idx(edge);
                    q = self.edge_quad_type(edge);
                    dx_wgt(edge_pt_idx) = (quad{q}.n/pi)...
                        *quad{q}.wgt.*self.dx_norm(edge_pt_idx);
                end
            end
        end
        
        %% 
        function T = get_total_t(self,quad)
            T = zeros(1,self.num_pts);
            for edge = 1:self.num_edges
                pt_idx = self.get_edge_point_idx(edge);
                q = self.edge_quad_type(edge);
                T(pt_idx) = quad{q}.t+2*pi*(edge-1);
            end
        end
        
        %%
        function plot_boundary(self,type)
            xmin = min(self.x);
            xmax = max(self.x);
            alpha = 0.1;
            Xmin = xmin(1) - alpha*(xmax(1)-xmin(1));
            Xmax = xmax(1) + alpha*(xmax(1)-xmin(1));
            Ymin = xmin(2) - alpha*(xmax(2)-xmin(2));
            Ymax = xmax(2) + alpha*(xmax(2)-xmin(2));
            hold on
            for seg = 1:self.num_segs
                pt_idx = self.get_seg_point_idx(seg);
                X = self.x(pt_idx,1);
                Y = self.x(pt_idx,2);
                switch type
                    case 'basic-'
                        plot([X;X(1)],[Y;Y(1)],'k-')
                    case 'basic-o'
                        plot([X;X(1)],[Y;Y(1)],'-o')
                    case 'quiver'
                        U = circshift(X,-1)-X;
                        V = circshift(Y,-1)-Y;
                        quiver(X,Y,U,V,0);
                    otherwise
                        error('plot type "%s" not recognized')
                end
            end
            hold off
            axis image
            axis([Xmin,Xmax,Ymin,Ymax])
            grid minor
            drawnow
        end
        
        %% 
        function plot_trace(self,type,fvals,LineSpec,quad)
            for seg = 1:self.num_segs
                if seg == 2
                    hold on
                end
                pt_idx = self.get_seg_point_idx(seg);
                T = self.get_total_t(quad);
                Tlen = 2*pi*numel(self.seg_edge_idx{seg});
                switch type
                    case 'plot'
                        X = pt_idx;             X = [X(:);X(end)+1];
                        Y = fvals(pt_idx);      Y = [Y(:);Y(1)];
                        plot(X,Y,LineSpec)
                    case 'log'
                        X = pt_idx;             X = [X(:);X(end)+1];
                        Y = abs(fvals(pt_idx)); Y = [Y(:);Y(1)];
                        semilogy(X,Y,LineSpec)
                    case 'plot-physical'
                        X = T(pt_idx);          X = [X(:);X(1)+Tlen];
                        Y = fvals(pt_idx);      Y = [Y(:);Y(1)];
                        plot(X,Y,LineSpec)
                    case 'log-physical'
                        X = T(pt_idx);          X = [X(:);X(1)+Tlen];
                        Y = abs(fvals(pt_idx)); Y = [Y(:);Y(1)];
                        semilogy(X,Y,LineSpec)
                    otherwise 
                        error('plot type "%s" not recognized',type)
                end
            end
            hold off
            if strcmp(type,'plot-physical') || strcmp(type,'log-physical')
                [tx,labels] = get_t_labels(self);
                xticks(tx);
                xticklabels(labels);
            end
            drawnow
        end
        
        %%
        function [tx,labels] = get_t_labels(self)
            tx = 0:2*pi:2*pi*self.num_edges;
            labels = cell(self.num_edges+1,1);
            for i = 1:self.num_edges+1
                labels{i} = sprintf('%d%s',2*(i-1),'\pi');
            end
        end
    end
end