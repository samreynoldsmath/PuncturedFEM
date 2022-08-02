%{
    asdf

%}

function [L2,H1] = InnerProductWithPolynomial(f,p,Q,z,elm,quad,D2N_args)
        
    %% Polynomial anti-Laplacians
    P = p.anti_laplacian_poly();
    P_trace = P.eval(elm.x(:,1),elm.x(:,2),z);
    Q_trace = Q.eval(elm.x(:,1),elm.x(:,2),z);

    Qstar = Q.anti_laplacian_poly();
    Qstar_trace = Qstar.eval(elm.x(:,1),elm.x(:,2),z);
    Qstar_nd_wgt = PolynomialWeightedNormalDerivative(Qstar,z,elm,quad);
    
    %% Traces of v-P and w-Q 
    fP_trace = f(:)-P_trace;
    
    %% Coefficients of polynomial product: P*Q
    PQ = P.prod_poly(Q);
    
    %% Coefficients of polynomial product: grad(P)*grad(Q)
    [Px,Py] = P.grad_poly();
    [Qx,Qy] = Q.grad_poly();
    PxQx = Px.prod_poly(Qx);
    PyQy = Py.prod_poly(Qy);
    gradP_gradQ = PxQx.sum_poly(PyQy);
    
    %% Dirichlet-to-Neumann map for harmonic functions
    vP_nd_wgt = Dirichlet2Neumann(fP_trace,elm,quad,D2N_args);
        
    %% Integrate
    
    % Integration weights for trapezoid rule
    h = pi/quad{1}.n; % for pre-weighted terms
    
    % Integrate v*w
    L2 = IntegratePolynomial(PQ,z,elm,quad)...
        + h*sum( ...
            + fP_trace.*Qstar_nd_wgt ...
            - vP_nd_wgt.*Qstar_trace ...
        );
    
    % Integrate \nabla v * \nabla w
    H1 = IntegratePolynomial(gradP_gradQ,z,elm,quad)...
        +h*sum( vP_nd_wgt.*Q_trace );
    
end

%% Integate a polynomial p over the element volume
function val = IntegratePolynomial(p,z,elm,quad)
    %{
        Eqn (9) \int_K (x-z)^\alpha dx = a boundary integral
    %}

    % precompute (x-z)*n
    xz = elm.x(:,1)-z(1);
    yz = elm.x(:,2)-z(2);
    xn = xz.*elm.unit_normal(:,1)+yz.*elm.unit_normal(:,2);
    
    % integrate each monomial term
    val = 0;
    for i = 1:p.nz
        alpha = p.multi_index(p.idx(i));
        xa = xz.^alpha(1);
        ya = yz.^alpha(2);
        f = xa.*ya.*xn/(2+alpha(1)+alpha(2));
        val = val + p.coef(i)*elm.integrate_over_boundary(f,quad);
    end
    
end

%% Obtain normal derivative of a polynomial
function nd_wgt = PolynomialWeightedNormalDerivative(p,z,elm,quad)

    % precompute
    xz=elm.x(:,1)-z(1);
    yz=elm.x(:,2)-z(2);
    
    nd=zeros(size(xz));
    
    % obtain coefficients of gradient
    [px,py]=p.grad_poly();
    
    % sum over coefficients of x component
    for i=1:px.nz
        alpha=px.multi_index(px.idx(i));
        xa=xz.^alpha(1);
        ya=yz.^alpha(2);
        nd=nd+px.coef(i)*xa.*ya.*elm.unit_normal(:,1);
    end
    
    % and the y component 
    for i=1:py.nz
        alpha=py.multi_index(py.idx(i));
        xa=xz.^alpha(1);
        ya=yz.^alpha(2);
        nd=nd+py.coef(i)*xa.*ya.*elm.unit_normal(:,2);
    end
    
    % multiply by weights
    dx_wgt = elm.get_dx_wgt(quad);
    nd_wgt = nd.*dx_wgt;
    
end
