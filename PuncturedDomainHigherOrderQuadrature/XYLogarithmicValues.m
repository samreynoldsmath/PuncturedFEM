function w_val = XYLogarithmicValues(A,B,x,y,b,elm)
    w_val = zeros(size(x));
    for seg = 2:elm.num_segs
        for i = 1:numel(x)
            xz = x(i)-elm.seg_int_pt(seg,1);
            yz = y(i)-elm.seg_int_pt(seg,2);
            xy2 = xz*xz+yz*yz;
            w_val(i) = w_val(i) + ...
                0.25*b(seg-1)*(A*xz+B*yz)*log(xy2);
        end
    end
end