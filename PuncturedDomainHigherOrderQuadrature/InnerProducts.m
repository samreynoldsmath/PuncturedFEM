%{
    IntegrateOverCurvedElement

    Computes the integrals

        \int_K vw dx  ~and~  \int_K \nabla v * \nabla w dx

    only in terms of boundary integrals, where K is a curvilinear polygon,
    and v,w are functions in H^1(K) satisfying

        \Delta v = p in K , v = f on \partial K , 
        \Delta w = q in K , w = g on \partial K .

    We assume p,q : K -> R are polynomials (of degree at most m-2) and 
    f,g : \partial K -> R are continuous, and when restricted to any edge
    of \partial K, it holds that f,g are traces of polynomials of degree 
    at most m.
    
    Method is adapted from "quadrature for Implicitly-defined Finite
    Element Functions on Curvilinear Polygons" by J. Ovall, S. Reynolds,
    in progress, tentative publication date 2021.

    The Dirichlet-to-Neumann map is adapted from "A High-order Method for
    Evaluating Derivatives of Harmonic Functions in Planar Domains" by J.
    Ovall, S. Reynolds, SISC 2018.

    INPUT:
        elm is an object of type class_elm, a custom data structure
            that contains relevant data about the curvilinear polygon K.
        quad is an object of type class_quad, which contains the quadture
            weights and parameter samplings on [0,2*pi] for the 
            parameterization of the edges of K.
        z is a point in R^2, as used in the shifted monomial basis,
            { (x-z)^alpha : alpha is a multi-index }
        p,q are sparse polynomial objects (type class_poly) that specify
            the coefficients of the Laplacians of v and w, repectively, in
            the shifted monomial basis
        f,g are vectors containing the values of the traces of v and w, 
            respectively, on the boundary \partial K. The length of each
            vector is equal to the number of sampled boundary points as
            specified by quad.

    OUTPUT:
        L2 is the inner product \int_K vw dx 
        H1 is the semi-inner product \int_K \nabla v * \nabla w dx

    POTENTIAL IMPROVEMENTS:
        * Compute coefficients of anti-Laplacians of polynomials offline
            (use a look-up table, for example)
        * Set up and solve Nystrom systems in parallel
        * Make option for when v = w

    Last updated 5/24/2022

%}

function [L2,H1] = InnerProducts(f,p,g,q,z,elm,quad,D2N_args,do_L2,do_H1)
        
    %% Polynomial anti-Laplacians
    P = p.anti_laplacian_poly();
    P_trace = P.eval(elm.x(:,1),elm.x(:,2),z);
    Q = q.anti_laplacian_poly();
    Q_trace = Q.eval(elm.x(:,1),elm.x(:,2),z);
    
    if do_L2
        Pstar = P.anti_laplacian_poly();
        Pstar_trace = Pstar.eval(elm.x(:,1),elm.x(:,2),z);
        Pstar_nd_wgt = PolynomialWeightedNormalDerivative(Pstar,z,elm,quad);
        
        Qstar = Q.anti_laplacian_poly();
        Qstar_trace = Qstar.eval(elm.x(:,1),elm.x(:,2),z);
        Qstar_nd_wgt = PolynomialWeightedNormalDerivative(Qstar,z,elm,quad);
    end
    
    %% Traces of v-P and w-Q 
    fP_trace = f(:)-P_trace;
    gQ_trace = g(:)-Q_trace;
    
    %% Coefficients of polynomial product: P*Q
    if do_L2
        PQ = P.prod_poly(Q);
    end
    
    %% Coefficients of polynomial product: grad(P)*grad(Q)
    if do_H1
        [Px,Py] = P.grad_poly();
        [Qx,Qy] = Q.grad_poly();
        PxQx = Px.prod_poly(Qx);
        PyQy = Py.prod_poly(Qy);
        gradP_gradQ = PxQx.sum_poly(PyQy);
    end
    
    %% Dirichlet-to-Neumann map for harmonic functions
    %{
        These normal derivatives, by construction, are "pre-weighted."
        See eqns (32) and (33) in "A High-order Method for Evaluating 
        Derivatives of Harmonic Functions in Planar Domains" by J. Ovall, 
        S. Reynolds, SISC 2018.
    %}
    wQ_nd_wgt = Dirichlet2Neumann(gQ_trace,elm,quad,D2N_args);
    vP_nd_wgt = Dirichlet2Neumann(fP_trace,elm,quad,D2N_args);
    
    %% Anti-Laplacian of v-P
    %{
        Given by Prop. 2.2 
    %}
    if do_L2
        [PHI,PHI_nd_wgt] = ...
            AntiLaplacianHarmonic(fP_trace,elm,quad,D2N_args);
    end
    
    %% Integrate
    
    % Integration weights for trapezoid rule
    h = pi/quad{1}.n; % for pre-weighted terms
    
    % Integrate v*w
    if do_L2
        L2 = IntegratePolynomial(PQ,z,elm,quad)...
            + h*sum( ...
                - (PHI+Pstar_trace).*wQ_nd_wgt ...
                - Qstar_trace.*vP_nd_wgt ...
                + (PHI_nd_wgt+Pstar_nd_wgt).*gQ_trace ...
                + Qstar_nd_wgt.*fP_trace ...
            );
    else
        L2 = NaN;
    end
    
    % Integrate \nabla v * \nabla w
    if do_H1
        H1 = IntegratePolynomial(gradP_gradQ,z,elm,quad)...
            +h*sum( vP_nd_wgt.*g(:) + wQ_nd_wgt.*P_trace );
    else
        H1 = NaN;
    end
    
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
