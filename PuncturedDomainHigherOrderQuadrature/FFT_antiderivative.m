function V = FFT_antiderivative(dVdt,elm)
    
    V = zeros(elm.num_pts,1);
    
    for seg = 1:elm.num_segs
        
        % get indices of points on segment
        pt_i = elm.get_seg_point_idx(seg);
        ni = numel(pt_i);
        num_edges_seg = numel(elm.seg_edge_idx{seg});
        
        % Fourier coefficients
        alpha = fft(dVdt(pt_i)); 

        % Differentiate with IFFT
        fft_idx = -num_edges_seg*1i./[0:(ni/2-1), 1, (-ni/2+1):(-1)]; 
        V(pt_i) = real( ifft( fft_idx(:).*alpha(:) ) );
        
    end

end