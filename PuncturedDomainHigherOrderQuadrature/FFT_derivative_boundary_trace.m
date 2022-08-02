function dvdt = FFT_derivative_boundary_trace(V,elm)
    
    dvdt = zeros(elm.num_pts,1);
    
    for seg = 1:elm.num_segs
        
        % get indices of points on segment
        pt_i = elm.get_seg_point_idx(seg);
        ni = numel(pt_i);
        num_edges_seg = numel(elm.seg_edge_idx{seg});
        
        % Fourier coefficients
        alpha = fft(V(pt_i)); 

        % Differentiate with IFFT
        fft_idx = [0:(ni/2-1), 0, (-ni/2+1):(-1)]/num_edges_seg; 
        dvdt(pt_i) = real( ifft( 1i*fft_idx(:).*alpha(:) ) );
        
    end
    
end