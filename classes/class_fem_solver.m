classdef class_fem_solver
    
    properties
        
        dof             % global degrees of freedom
        B               % global FEM operator (linear combination of 
                            % stiffness and mass matrices)
        rhs             % global right-hand side vector
        B_loc
        rhs_loc
        dir_idx         % indices of global functions with Dirichlet BC
        neu_idx         % indices of global functions with Neumann BC
        
    end
    
    methods
        
        %%
        function self = init(self,mesh,B_loc,rhs_loc)
            
            self.dof = mesh.num_glob_funs;
            self.B_loc = B_loc;
            self.rhs_loc = rhs_loc;
        
        end
        
        %%
        function coef = solve(self,gmres_args)
            coef = gmres(self.B,self.rhs,...
                gmres_args.restart,...
                gmres_args.tol,...
                gmres_args.maxit );
        end
        
        %%
        function self = assemble(self,mesh)
            
            self.B = zeros(self.dof,self.dof);
            self.rhs = zeros(self.dof,1);
            
            for i = 1:mesh.num_glob_funs
                % matrix operator
                for j = 1:mesh.num_glob_funs
                    for k = intersect(mesh.supp{i},mesh.supp{j})
                        v = mesh.loc_funs{i}(mesh.supp{i} == k);
                        w = mesh.loc_funs{j}(mesh.supp{j} == k);
                        self.B(i,j) = self.B(i,j) + self.B_loc(v,w);
                    end
                end
                % rhs vector
                for k = mesh.supp{i}
                    v = mesh.loc_funs{i}(mesh.supp{i} == k);
                    self.rhs(i) = self.rhs(i) + self.rhs_loc(v);
                end
            end
            
            % impose boundary conditions
            for i = self.dir_idx
                self.B(i,:) = 0;
                self.B(i,i) = 1;
                self.rhs(i) = 0;
            end
            
        end
        
        %%
        
    end
    
end