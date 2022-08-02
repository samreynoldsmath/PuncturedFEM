function [gx,gy] = XYLogarithmicGradient(A,B,b,elm)
    
    gx = zeros(elm.num_pts,1);
    gy = zeros(elm.num_pts,1);
    
    for seg = 2:elm.num_segs
        for i = 1:elm.num_pts
            xz = elm.x(i,1)-elm.seg_int_pt(seg,1);
            yz = elm.x(i,1)-elm.seg_int_pt(seg,2);
            AxBy = A*xz+B*yz;
            xy2 = xz*xz+yz*yz;
            gx(i) = gx(i) + b(seg-1)*(0.5*(AxBy/xy2)*xz+0.25*A*log(xy2));
            gy(i) = gy(i) + b(seg-1)*(0.5*(AxBy/xy2)*yz+0.25*B*log(xy2));
        end
    end
    
end