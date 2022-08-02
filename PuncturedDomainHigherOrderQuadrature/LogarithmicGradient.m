function [gx,gy] = LogarithmicGradient(a,elm)
    
    gx = zeros(elm.num_pts,1);
    gy = zeros(elm.num_pts,1);
    
    for seg = 2:elm.num_segs
        for i = 1:elm.num_pts
            xz = elm.x(i,1)-elm.seg_int_pt(seg,1);
            yz = elm.x(i,2)-elm.seg_int_pt(seg,2);
            xy2 = xz*xz+yz*yz;
            gx(i) = gx(i) + a(seg-1)*xz/xy2;
            gy(i) = gy(i) + a(seg-1)*yz/xy2;
        end
    end
    
end