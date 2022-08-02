function u_nd_wgt = Dirichlet2Neumann(u,elm,quad,D2N_args)
    %{
        Given the tangential derivative u_td of a harmonic function u, 
        returns the normal derivative u_nd. This is accomplished by 
        solving an integral equation via a Nystrom method to obtain the 
        trace of a harmonic conjuate v on the boundary, and then 
        differentiating with respect to the boundary parameter t to obtain 
        the tangential derivative of v. Since v is a harmonic conjugate of 
        u, we have that v_td = u_nd.
    %}
    
    % Trace of a harmonic conjugate of U = u - sum_k a_k*log|x-z_k|
    [V,a] = HarmonicConjugateTrace(u,elm,quad,D2N_args);
    
    % Differentiate the harmonic conjugate to get the tangential derivative
    V_td_wgt = FFT_derivative_boundary_trace(V,elm);
    
    % Get the tangential derivative of w = sum_k a_k*log|x-z_k|
    w_nd_wgt = LogarithmicWeightedNormalDerivative(a,elm,quad);
    
    % Normal derivative of u = U + w
    u_nd_wgt = V_td_wgt+w_nd_wgt;
    
end