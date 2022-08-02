function [W,W_nd] = LogarithmicAntiLaplacian(elm,a)
    W = zeros(elm.num_pts,1);
    W_nd = zeros(elm.num_pts,1);
    for seg = 2:elm.num_segs
        for i = 1:elm.num_pts
            xz = elm.x(i,:)-elm.seg_int_pt(seg,:);
            [W0,Wx,Wy] = T(xz(1),xz(2));
            W(i) = W(i) + a(seg-1)*W0;
            W_nd(i) = W_nd(i) + a(seg-1)*(...
                Wx*elm.unit_normal(i,1) + ...
                Wy*elm.unit_normal(i,2) );
        end
    end
end

function [W,Wx,Wy] = T(x,y)
    xy2 = x*x+y*y;
    logxy2 = log(xy2);
    atat = atan(abs(x/y))+atan(abs(y/x));
    W = xy2*(0.5*logxy2-1)+x*y*atat;
    Wx = x*(logxy2-1)+y*atat;
    Wy = y*(logxy2-1)+x*atat;
end