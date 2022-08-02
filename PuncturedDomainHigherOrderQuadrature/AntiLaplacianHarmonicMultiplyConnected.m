function [PHI,PHI_nd_wgt] = ...
    AntiLaplacianHarmonicMultiplyConnected(phi,elm,quad,D2N_args)

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
    [psi_hat,a] = HarmonicConjugateTrace(phi,elm,quad,D2N_args);
        
    % obtain the trace of psi
    psi = phi - LogarithmicTraces(elm,a);
        
    % anti-Laplacian of w = sum_k a_k*log|x-z_k|
    [W,W_nd] = LogarithmicAntiLaplacian(elm,a);
    
    % get trace of rho from its tangential derivative
    rho_nd = ...
        + psi.*elm.unit_normal(:,1) ...
        - psi_hat.*elm.unit_normal(:,2);
    
    % check existence of rho
    rho_exists = check_existence_nuemann_problem(rho_nd,elm,quad);
    
    if rho_exists    
        %% case 1: rho exists
        %{
            It holds that rho is harmonic. Write
                rho = eta + sum_k b_k*ln|x-z_k|
            where eta has a harmonic conjugate.
        %}
        
        rho = SolveNeumannProblem(rho_nd,elm,quad);
%         rho = 0.5*log(elm.x(:,1).^2+elm.x(:,2).^2);
        
        % get a harmonic conjugate of eta and logarithmic coefficients
        [eta_hat,b] = HarmonicConjugateTrace(rho,elm,quad,D2N_args);

        % obtain the trace of eta
        eta = rho - LogarithmicTraces(elm,b);
        
        % anti-Laplacian of Z = sum_k b_k*(A*x+B*y)/(x^2+y^2)
        A = 1;
        B = 0;
        Z = XYLogarithmicTraces(A,B,elm,b);
        
        % gradient and normal derivative
        Z_nd = XYLogarithmicNormalDerivative(A,B,b,elm);
        
        % anti-Laplacian of phi
        PHI = 0.25*(elm.x(:,1).*eta+elm.x(:,2).*eta_hat) + Z + W;
                
        % weighted normal derivative of PHI
        [gx,gy] = LogarithmicGradient(b,elm);
        PSI_nd = ...
            0.25*elm.unit_normal(:,1).*( ...
                eta ...
                + elm.x(:,1).*(psi-gx) ...
                + elm.x(:,2).*(psi_hat+gy) ...
            ) + ...
            0.25*elm.unit_normal(:,2).*( ...
                eta_hat ...
                + elm.x(:,2).*(psi-gx) ...
                - elm.x(:,1).*(psi_hat+gy) ...
            );
        dx_wgt = elm.get_dx_wgt(quad);
        PHI_nd_wgt = ( PSI_nd + Z_nd + W_nd ).*dx_wgt;
        
    else
        %% case 2: rhohat exists
        %{
            CONJECTURE:
            It holds that there is a solution to
                grad(rhohat) = (psihat,psi)
            and that rhohat is harmonic. Write
                rho = eta + sum_k b_k*ln|x-z_k|
            where eta has a harmonic conjugate.
        %}
    
        % get trace of rho-hat from its tangential derivative 
        rho_hat_nd = ...
            + psi_hat.*elm.unit_normal(:,1) ...
            + psi.*elm.unit_normal(:,2);
        rho_hat = SolveNeumannProblem(rho_hat_nd,elm,quad);
        
        % get a harmonic conjugate of eta and logarithmic coefficients
        [theta_hat,c] = HarmonicConjugateTrace(rho_hat,elm,quad,D2N_args);

        % obtain the trace of eta
        theta = rho - LogarithmicTraces(elm,c);
        
        % anti-Laplacian of Z = sum_k b_k*(A*x+B*y)/(x^2+y^2)
        A = 0;
        B = 1;
        Z = XYLogarithmicTraces(A,B,elm,c);
        
        % gradient and normal derivative
        Z_nd = XYLogarithmicNormalDerivative(A,B,c,elm);
        
        % anti-Laplacian of phi
        PHI = 0.25*(elm.x(:,2).*theta-elm.x(:,1).*theta_hat) + Z + W;
                
        % weighted normal derivative of PHI
        [gx,gy] = LogarithmicGradient(c,elm);
        PSI_nd = ...
            0.25*elm.unit_normal(:,1).*( ...
                -theta_hat ...
                + elm.x(:,2).*(psi_hat-gx) ...
                + elm.x(:,1).*(psi-gy) ...
            ) + ...
            0.25*elm.unit_normal(:,2).*( ...
                theta ...
                - elm.x(:,1).*(psi_hat-gx) ...
                + elm.x(:,2).*(psi-gy) ...
            );
        dx_wgt = elm.get_dx_wgt(quad);
        PHI_nd_wgt = ( PSI_nd + Z_nd + W_nd ).*dx_wgt;
        
    end
        
end