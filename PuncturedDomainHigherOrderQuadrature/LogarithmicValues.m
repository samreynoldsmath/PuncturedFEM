function w_val = LogarithmicValues(x,y,a,elm)
    w_val = zeros(size(x));
    for seg = 2:elm.num_segs
        for i = 1:numel(x)
            xx = x(i)-elm.seg_int_pt(seg,1);
            yy = y(i)-elm.seg_int_pt(seg,2);
            w_val(i) = w_val(i) + a(seg-1)*log(norm([xx,yy]));
        end
    end
end