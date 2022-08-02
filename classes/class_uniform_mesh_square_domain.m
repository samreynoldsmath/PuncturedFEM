classdef class_uniform_mesh_square_domain
    
    properties
    
        %% physical properties of mesh
        r                   % unit square domain partitioned as r x r grid 
                                % of cells with square outer boundary and 
                                % possibly holes
        num_verts           % number of vertices
        num_edges           % number of edges 
        num_closed_edges    % number of segments with a single edge
        num_cells           % number of cells
        
        %% boundary indices
        outer_boundary_idx  % indices for outer boundary
        inner_boundary_idx  % indices for inner boundary
        
        %% global functions
        num_glob_funs       % number of global functions
        supp                % ith entry has indices of cells on which 
                                % the ith global function is supported
        funs_on_cell        % kth entry has indices of global functions
                                % supported on kth cell
        type                % ith entry is a label of:
                                % 0: vertex function
                                % 1: edge function
                                % -1: support of the trace is a closed loop
                                % 2: bubble function
        loc_funs            % ith entry is an array of indices of local 
                                % functions [v_1,...] such that the ith
                                % global function is v_j when restricted
                                % to the jth cell listed in supp{i}
%         loc_pos             % ith entry is the (local) index list of the
%                                 % vertex/edge/volume associated with 
%                                 % the local functions in loc_fun{i}
    
    end
    
    methods
        
        %% initialization
        function self = init(self,r,elm,vmk)
            
            % physical properties of mesh
            self.r = r;
            self.num_cells = self.r^2;
            num_interior_edges = (elm.num_edges-4)*self.num_cells; 
            self.num_verts = (self.r+1)^2+num_interior_edges;
            self.num_edges = 2*self.r*(self.r+1)+num_interior_edges;
            self.num_closed_edges = self.num_cells...
                *elm.get_num_closed_edges;
            
            % get number of global functions
            self = self.get_num_glob_funs(vmk);
            
            % 
            self = self.get_support_data(vmk);
            
        end
        
        %% draw mesh
        function draw(self,elm)
            hold on
            h = 1/self.r;
            % edges
            x = h*elm.x(:,1);
            y = h*elm.x(:,2);
            % vertices            
            zx = x(elm.edge_idx(1:end-1));
            zy = y(elm.edge_idx(1:end-1));
            % don't mark vertices for edges that are closed loops
            closed_edges = elm.get_closed_edge_idx();
            zx(closed_edges) = [];
            zy(closed_edges) = [];
            % loop over cells in mesh
            for i = 1:self.r
                shift_x = (i-1)*h;
                for j = 1:self.r
                    shift_y = (j-1)*h;
                    for seg = 1:elm.num_segs
                        % indices
                        idx = elm.get_seg_point_idx(seg);
                        idx = [idx,idx(1)];
                        % plot edges
                        plot( x(idx)+shift_x, y(idx)+shift_y, 'k-' )
                    end
                    % plot vertices
                    scatter(zx+shift_x,zy+shift_y,[],'ko','filled')
                end
            end
            hold off
            axis image
            axis([-0.1,1.1,-0.1,1.1])
            grid minor
        end
        
        %% 
        function self = get_num_glob_funs(self,vmk)
            
            bottom_efun_idx = and(vmk.pos == 1,vmk.type == 1);
            num_top_bottom_efun = numel(find(bottom_efun_idx));
            
            right_efun_idx = and(vmk.pos == 2,vmk.type);
            num_left_right_efun = numel(find(right_efun_idx));
            
            num_efun = num_top_bottom_efun+num_left_right_efun;
            
            num_glob_vfun = self.num_verts-self.num_closed_edges;
            num_glob_efun = ...
                num_efun*(self.num_edges-self.num_closed_edges)/2;
            num_glob_pfun = vmk.num_pfun*self.num_cells;
            num_glob_bfun = vmk.num_bfun*self.num_cells;
            
            self.num_glob_funs = ...
                + num_glob_vfun ... 
                + num_glob_efun ...
                + num_glob_pfun ...
                + num_glob_bfun;
            
        end
        
        %% 
        function k = get_cell_idx_from_coord(self,i,j)
            k = (i-1)*self.r+j;
        end
        
        %%
        function [i,j] = get_cell_coord_from_idx(self,k)
            j = 1+mod(k-1,self.r);
            i = 1+(k-j)/self.r;
        end
        
        %%
        function self = get_support_data(self,vmk)
            
            % allocations
            self.funs_on_cell = cell(1,self.num_cells);
            self.supp = cell(1,self.num_glob_funs);
            self.type = zeros(1,self.num_glob_funs);
            self.loc_funs = cell(1,self.num_glob_funs);
%             self.loc_pos = cell(1,self.num_glob_funs);
            
            on_outer_boundary = false(1,self.num_glob_funs);
            on_inner_boundary = false(1,self.num_glob_funs);
            
            % track which local functions have been accounted for
            loc_funs_done = cell(1,self.num_cells);
            
            % index of global function
            i = 0;
            
            % loop over cells
            for k = 1:self.num_cells
                
                % loop over local functions on this cell
                for v = 1:vmk.dim
                    
                    % check if v has already been assigned to a global fun
                    is_new = self.check_if_fun_is_new(k,v,loc_funs_done);
                    
                    if is_new
                        
                        % define a new global function phi_i
                        i = i+1;
                        
                        % get type label (vertex,edge,loop,bubble)
                        self.type(i) = vmk.type(v);
                        
                        % find sisters of local function v
                        sis = vmk.sis_fun{v};
                        
                        % number of sister functions
                        num_sis = numel(sis);
                        
                        % assign v and its sisters to phi_i
                        self.loc_funs{i} = [v,sis];
                        
                        % get local position data
%                         self.loc_pos{i} = vmk.pos(self.loc_funs{i});
                        
                        % find mesh cells on which sisters are supported
                        % (neglect some sisters if on global boundary)
                        self.supp{i} = zeros(1,num_sis+1);
                        self.supp{i}(1) = k;
                        for j = 2:num_sis+1
                            s = sis(j-1);
                            self.supp{i}(j) = ...
                                self.find_supp_sister(vmk,k,v,s);
                        end
                        
                        % remove entries for cells on global boundary
                        cell_not_exist = ( self.supp{i} < 0 );
                        self.supp{i}(cell_not_exist) = [];
                        self.loc_funs{i}(cell_not_exist) = [];
%                         self.loc_pos{i}(cell_not_exist) = [];
                        
                        % mark phi_i as being supported on sister cells
                        for K_j = self.supp{i}
                            self.funs_on_cell{K_j} = ...
                                [self.funs_on_cell{K_j},i];
                        end
                        
                        % determine if trace of phi_i is supported on
                        % global boundary
                        if self.type(i) == 0 || self.type(i) == 1
                            % vertex or edge function
                            on_outer_boundary(i) = ...
                                ( 0 < numel(find(cell_not_exist)) );
                        elseif self.type(i) == -1
                            % loop function
                            % SPECIFIC TO PEGBOARD MESH
                            on_inner_boundary(i) = true; 
                        end
                        
                        % remove v and its sisters from later consideration
                        % for each cell K_j on which phi_i is supported
                        for j = 1:numel(self.supp{i})
                            % add the jth local function to the list 
                            % of completed functions on K_j
                            loc_funs_done{self.supp{i}(j)} = ...
                                [ loc_funs_done{self.supp{i}(j)},...
                                self.loc_funs{i}(j) ];
%                             loc_funs_done{j} = ...
%                                 [loc_funs_done{j},self.loc_funs{i}(j)];
                        end
                        
                    end
                    
                end
                
            end
            
            self.outer_boundary_idx = find(on_outer_boundary);
            self.inner_boundary_idx = find(on_inner_boundary);
            
        end
        
        %%
        function is_new = check_if_fun_is_new(self,k,v,loc_funs_done)
            
            % start by assuming v has not been completed
            is_new = true;
            
            % check if v is in the list of completed functions
            for w = loc_funs_done{k}
                if v == w
                    is_new = false;
                end
            end
            
        end
        
        %%
        function K_j = find_supp_sister(self,vmk,k,v,s)
            %{
                SPECIFIC TO PEGBOARD MESH
            
                Given a sister function w_j to v, where v is supported
                on the cell K (with index k), locate the adjacent cell
                K_j on which w_j is supported.
            
                If v is on the global boundary, so that w_j does not 
                exist, we set K_j = -1;
            %}
            v_pos = vmk.pos(v);
            
            % case break down for type
            switch vmk.type(v)
                case 0 % vertex function
                    s_pos = vmk.pos(s);
                    K_j = self.find_supp_sister_vertex(k,v_pos,s_pos);
                case 1 % edge function
                    K_j = self.find_supp_sister_edge(k,v_pos);
                case -1 % loop function
                    K_j = -1; % SPECIFIC TO PEGBOARD MESH
                case 2 % bubble function
                    K_j = -1; % bubble function supported on one cell only
            end
            
        end
        
        %%
        function is_boundary_cell = check_if_cell_on_boundary(self,k)
            
            [i,j] = self.get_cell_coord_from_idx(k);
            is_boundary_cell = (...
                i == 1 || ...
                i == self.r ||...
                j == 1 ||...
                j == self.r );
                
        end
        
        %%
        function K_j = find_supp_sister_vertex(self,k,v_pos,s_pos)
            
            % get cell coordinates
            [i,j] = get_cell_coord_from_idx(self,k);
            
            switch v_pos
                case 1 % v_pos
                    switch s_pos
                        case 2 % s_pos
                            if j == 1
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i,j-1);
                            end
                        case 3 % s_pos
                            if j == 1 || i == 1
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i-1,j-1);
                            end
                        case 4 % s_pos
                            if i == 1
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i-1,j);
                            end
                    end
                case 2 % v_pos
                    switch s_pos
                        case 1 % s_pos
                            if j == self.r
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i,j+1);
                            end
                        case 3 % s_pos
                            if i == 1
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i-1,j);
                            end
                        case 4 % s_pos
                            if i == 1 || j == self.r
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i-1,j+1);
                            end
                    end
                case 3 % v_pos
                    switch s_pos
                        case 1 % s_pos
                            if i == self.r || j == self.r 
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i+1,j+1);
                            end
                        case 2 % s_pos
                            if i == self.r
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i+1,j);
                            end
                        case 4 % s_pos
                            if j == self.r
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i,j+1);
                            end
                    end
                case 4 % v_pos
                    switch s_pos
                        case 1 % s_pos
                            if i == self.r
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i+1,j);
                            end
                        case 2 % s_pos
                            if i == self.r || j == 1
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i+1,j-1);
                            end
                        case 3 % s_pos
                            if j == 1
                                K_j = -1;
                            else
                                K_j = self.get_cell_idx_from_coord(i,j-1);
                            end
                    end
            end
            
            
        end
        
        %%
        function K_j = find_supp_sister_edge(self,k,v_pos)
            
            % get cell coordinates
            [i,j] = self.get_cell_coord_from_idx(k);
            
            is_boundary_edge = (...
                (i == 1 && v_pos == 1) || ...
                (i == self.r && v_pos == 3) || ...
                (j == 1 && v_pos == 4) || ...
                (j == self.r && v_pos == 2) );
            
            if is_boundary_edge
                K_j = -1;
            else
                switch v_pos
                    case 1
                        K_j = self.get_cell_idx_from_coord(i-1,j);
                    case 2
                        K_j = self.get_cell_idx_from_coord(i,j+1);
                    case 3
                        K_j = self.get_cell_idx_from_coord(i+1,j);
                    case 4
                        K_j = self.get_cell_idx_from_coord(i,j-1);
                end
            end
            
        end
        
        %%
        function plot_linear_combo(self,intval,coef,fig_type,contour_lines)
            
            Nx = self.r*intval.n_x;
            Ny = self.r*intval.n_y;
            
            X = zeros(Nx,Ny);
            Y = zeros(Nx,Ny);
            Z = zeros(Nx,Ny);
            
            h = 1/self.r;
            for i = 1:self.r
                shift_y = (i-1)*h;
                ii = (i-1)*intval.n_y + (1:intval.n_y);
                for j = 1:self.r
                    shift_x = (j-1)*h;
                    jj = (j-1)*intval.n_x + (1:intval.n_x);
                    
                    % physical points in domain
                    X(ii,jj) = shift_x + h*intval.x;
                    Y(ii,jj) = shift_y + h*intval.y;
                    
                    % map coefficients to local functions
                    k = get_cell_idx_from_coord(self,i,j);

                    c = zeros(1,intval.num_funcs);
                    for phi = self.funs_on_cell{k}
                        for v = self.loc_funs{phi}(self.supp{phi} == k)
                            c(v) = c(v) + coef(phi);
                        end
                    end

                    for v = 1:intval.num_funcs
                        Z_loc = reshape(...
                            c(v)*intval.val(:,v),...
                            intval.n_x,intval.n_y );
                        Z(ii,jj) = Z(ii,jj) + Z_loc;
                    end
                    
                end
            end
            
            % make plot
            switch fig_type
                case 'surf'
                    surf(X,Y,Z,'EdgeColor','none')
                case 'contour'
                    contour(X,Y,Z,contour_lines)
                otherwise
                    error('fig_type = "%s" not recognized',fig_type)
            end
            axis image
            axis off
            
        end
        
        %%
        
    end
end