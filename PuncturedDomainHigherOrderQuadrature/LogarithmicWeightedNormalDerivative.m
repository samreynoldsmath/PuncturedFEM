function dwdn_wgt = LogarithmicWeightedNormalDerivative(a,elm,quad)
    
    dx_wgt = elm.get_dx_wgt(quad);
    dwdn_wgt = zeros(elm.num_pts,1);
    
    [gx,gy] = LogarithmicGradient(a,elm);
    
    for i = 1:elm.num_pts
        dwdn_wgt(i) = dx_wgt(i)...
            *(gx(i)*elm.unit_normal(i,1)+gy(i)*elm.unit_normal(i,2));
    end
    
end