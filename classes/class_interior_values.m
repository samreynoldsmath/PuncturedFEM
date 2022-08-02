classdef class_interior_values
    
    properties 
        
        x_min
        x_max
        y_min
        y_max
        n_x
        n_y
        n
        x
        y
        in
        tol
        num_funcs
        val
        
    end
    
    methods
        
        %% 
        function self = init(self,elm,args)
            
            self.num_funcs = args.num_funcs;
            self.n_x = args.n_x;
            self.n_y = args.n_y;
            self.tol = args.tol;
            self.n = self.n_x*self.n_y;
            
            self = self.get_bounds(elm.x(:,1),elm.x(:,2));
            
        end
        
        %%
        function self = set_sizes(self)
            [self.n_x,self.n_y] = size(self.x);
            self.n = self.n_x*self.n_y;
        end
        
        %%
        function self = get_bounds(self,x,y)
            self.x_min = min(x);
            self.x_max = max(x);
            self.y_min = min(y);
            self.y_max = max(y);
        end
        
        %%
        function self = get_physical_pts(self,elm)
                        
            %%
            X = self.x_min:(1/(self.n_x-1)):self.x_max;
            Y = self.y_min:(1/(self.n_y-1)):self.y_max;
            [self.x,self.y] = meshgrid(X,Y);
            
            %% determine with points lie strictly within the boundary
            self.in = self.is_inside(elm);
            
        end
        
        %%
        function in = is_inside(self,elm)
            %{
                inpolygon supports non-convex and self-intersecting polygons.
                The function also supports multiply-connected or disjoint polygons; 
                however, the distinct edge loops should be separated by NaNs. In the
                case of multiply-connected polygons, the external and internal loops
                should have opposite orientations; for example, a counterclockwise 
                outer loop and clockwise inner loops or vice versa. Self-intersections
                are not supported in this context due to the ambiguity associated with
                loop orientations.
            %}
            
            % find points in strict interior
            Y = zeros(elm.num_pts+2*elm.num_segs-1,2);
            for seg = 1:elm.num_segs
                pt_idx = elm.get_seg_point_idx(seg);
                Y(pt_idx+2*(seg-1),:) = ...
                    elm.x(pt_idx,:) - ...
                    self.tol*elm.unit_normal(pt_idx,:);
                Y(pt_idx(end)+2*seg-1,:) = Y(pt_idx(1)+2*(seg-1),:);
                if seg < elm.num_segs
                    Y(pt_idx(end)+2*seg,:) = NaN;
                end
            end
            in = inpolygon(self.x,self.y,Y(:,1),Y(:,2));

            % disqualify points near corners
            for seg = 1:elm.num_segs
                seg_corner_idx = elm.seg_edge_idx{seg};
                seg_num_pts = elm.seg_idx(seg+1)-elm.seg_idx(seg);
                for i = elm.edge_idx(seg_corner_idx)
                    k = mod(i-2,seg_num_pts)+1;
                    x1 = elm.dx(i,:);
                    x2 = elm.dx(k,:);
                    xx = dot(x1,x2);
                    
                    if seg == 1
                        s = 1;
                    else
                        s = -1;
                    end
                    
                    if s*xx <= 0
                        th = acos(xx/(norm(x1)*norm(x2)));
                        phi = 0.5*(pi-th);
                        R2 = self.tol/sin(phi);
                        R2 = R2*R2;
                    else
                        R2 = self.tol*self.tol;
                    end
                    for j = 1:self.n
                        sq_dist = ...
                            (elm.x(i,1)-self.x(j))^2 + ...
                            (elm.x(i,2)-self.y(j))^2;
                        if sq_dist <= R2
                            in(j) = false;
                        end
                    end
                end
            end

        end
        
        %%
        function show_points(self,elm)
            scatter(self.x(self.in),self.y(self.in),[],'bo','filled')
            hold on
            elm.plot_boundary('basic-o')
            hold off
        end
        
        %%
        function self = get_int_vals(self,vmk,elm,quad,D2N_args)
            
            self.val = zeros(self.n,self.num_funcs);
            z = [0,0];
            
            for k = 1:vmk.dim
                
                fprintf('Obtaining interior values: %4d / %4d\n',...
                    k,vmk.dim);
                
                P = vmk.p{k}.anti_laplacian_poly();
                P_trace = P.eval(elm.x(:,1),elm.x(:,2),z);
                fP_trace = vmk.f(:,k)-P_trace(:);
                val_grid = ...
                    P.eval(self.x,self.y,z) + ...
                    cauchy_integral_harmonic_function(self,...
                        fP_trace,elm,quad,D2N_args) ;
                self.val(:,k) = val_grid(:);
            end
        end
        
        %%
        function u = cauchy_integral_harmonic_function(self,...
                f,elm,quad,d2n_args)
            %{
                f is vector of values of Dirichlet trace of u
                
                write u = U + sum_k a_k*log|x-z_k|
                find a_k's and V: harmonic conjugate of U
            %}
            
            [V,a] = HarmonicConjugateTrace(f,elm,quad,d2n_args);
            u = LogarithmicValues(self.x,self.y,a,elm);
            w = LogarithmicTraces(elm,a);
            U = f(:)-w(:);
            
            for edge = 1:elm.num_edges
                q = elm.edge_quad_type(edge);
                wgt = quad{q}.wgt;
                pt_idx = elm.get_edge_point_idx(edge);
                for j = 1:numel(pt_idx)
                    jj = pt_idx(j);
                    W = ( U(jj)+1i*V(jj) )/(2i*pi);
                    dz = elm.dx(jj,1)+1i*elm.dx(jj,2);
                    numer = W*dz*wgt(j);
                    z = elm.x(jj,1)+1i*elm.x(jj,2);
                    for i = 1:self.n
                        if self.in(i)
                            z0 = self.x(i)+1i*self.y(i);
                            denom = z-z0;
                            u(i) = u(i) + real(numer/denom);
                        else 
                            u(i) = NaN;
                        end
                    end
                end
            end
            
        end
        
        %%
        function plot_v(self,k,elm,fig_type,contour_lines,make_colorbar)
            
            val_grid = reshape(self.val(:,k),self.n_x,self.n_y);
            
            switch fig_type
                case 'contour'
                    contour(self.x, self.y, val_grid, contour_lines)
                    if make_colorbar
                        colorbar
                    end
                    hold on
                    elm.plot_boundary('basic-')
                    hold off
                    grid off
                    axis tight
                    axis off
                case 'surf'
                    surf(self.x, self.y, val_grid,'EdgeColor','interp')
                otherwise
                    error('fig_type = "%s" not recognized',fig_type)
            end
            
            drawnow
            
        end
        
        %%
        function plot_all(self,fig_num,elm,fig_type,contour_lines,...
                make_colorbar)
            for k = 1:self.num_funcs
                figure(fig_num+k-1)
                self.plot_v(k,elm,fig_type,contour_lines,make_colorbar)
            end
        end
        
        %%
        
        
    end
    
end


