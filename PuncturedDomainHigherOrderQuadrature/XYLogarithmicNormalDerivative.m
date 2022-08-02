function dZ_nd = XYLogarithmicNormalDerivative(A,B,b,elm)

    dZ_nd = zeros(elm.num_pts,1);
    
    [gx,gy] = XYLogarithmicGradient(A,B,b,elm);
    
    for i = 1:elm.num_pts
        dZ_nd(i) = dZ_nd(i) + (...
            gx(i)*elm.unit_normal(i,1) + ...
            gy(i)*elm.unit_normal(i,2) );
    end
    
end