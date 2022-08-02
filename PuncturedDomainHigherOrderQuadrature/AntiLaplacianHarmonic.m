function [PHI,PHI_nd_wgt] = AntiLaplacianHarmonic(phi,elm,quad,D2N_args)

    if elm.num_segs == 1 
        % simply connected
        [PHI,PHI_nd_wgt] = ...
            AntiLaplacianHarmonicSimplyConnected(phi,elm,quad,D2N_args);
    else
        % multiply connected
        [PHI,PHI_nd_wgt] = ...
            AntiLaplacianHarmonicMultiplyConnected(phi,elm,quad,D2N_args);
    end
    
end