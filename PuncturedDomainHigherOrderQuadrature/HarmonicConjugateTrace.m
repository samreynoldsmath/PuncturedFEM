function [V,a] = HarmonicConjugateTrace(boundary_trace,elm,quad,args)
    switch args.type
        case 'density'
            [V,a] = HarmonicConjugateFromDoubleLayerDensity(...
                boundary_trace,elm,quad,args);
        case 'direct'
            error("functionality not yet supported");
%             [V,a] = HarmonicConjugateDirectSolve(...
%                 boundary_trace,elm,quad,args);
        otherwise
            error('D2N map "%s" not recognized',type);
    end
end

