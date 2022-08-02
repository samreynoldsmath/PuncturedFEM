function [PHI,PHI_nd_wgt] = ...
    AntiLaplacianHarmonicSimplyConnected(phi,elm,quad,D2N_args)

    %{
      Arguments:
        phi: a harmonic function
        elm: object of type class_element
        quad: cell array of objects of type class_quad
            quad{1}: trapezoid rule
            quad{2}: Kress quadrature
            quad{3}: Martensen quadrature (if necessary)
        D2N_args: struct containing parameters for Dirichlet-to-Neumann map
            type: string specifying the D2N map
            restart,tol,maxit: parameters for GMRES
        
      Returns: trace of PHI, and its weighted normal derivative, such that
            the Laplacian of PHI is phi
        
        Assumes that the domain is simply connected.
        See Prop. 2.2 in "Quadrature for Implicitly-Defined Finite Element
        Functions on Curvilinear Polygons"
    %}

    % compute harmonic conjugate of psi = phi - sum_k a_k*log|x-z_k|
    [phihat,~] = HarmonicConjugateTrace(phi,elm,quad,D2N_args);
    
    % get trace of rho from its tangential derivative
    rho_td = phi.*elm.unit_tangent(:,1)-phihat.*elm.unit_tangent(:,2);
    rho = TraceFromTangentialDerivative(rho_td,elm,quad);
    
    % get trace of rho-hat from its tangential derivative 
    rhohat_td = phihat.*elm.unit_tangent(:,1)+phi.*elm.unit_tangent(:,2);
    rhohat = TraceFromTangentialDerivative(rhohat_td,elm,quad);
    
    % compute trace of PHI and its normal derivative
    PHI = 0.25*(elm.x(:,1).*rho+elm.x(:,2).*rhohat);
    PHI_nd = ...
        0.25*(rho+elm.x(:,1).*phi+elm.x(:,2).*phihat)...
        .*elm.unit_normal(:,1) + ...
        0.25*(rhohat-elm.x(:,1).*phihat+elm.x(:,2).*phi)...
        .*elm.unit_normal(:,2);
    
    % weighted normal derivative of PHI
    dx_wgt = elm.get_dx_wgt(quad);
    PHI_nd_wgt = PHI_nd.*dx_wgt;
        
end