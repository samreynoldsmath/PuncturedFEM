function rho_exists = check_existence_nuemann_problem(rho_nd,elm,quad)
    %{
        See Theorem 5.2 in "On the Dirichlet and the Neumann problems for 
        Laplace equation in multiply connected domains" 
        by A. Cialdea, V. Leonessa and A. Malaspina; 
        Complex Variables and Elliptic Equations (2010). 
    %}
    tol = 1e-8;
    rho_exists = true;
    for seg = 1:elm.num_segs
        int_seg_rho_nd = elm.integrate_over_seg(rho_nd,seg,quad);
        rho_exists = rho_exists && ( int_seg_rho_nd < tol );
    end
    
end