function u = TraceFromTangentialDerivative(dudt,elm,quad)
    dx_wgt = elm.get_dx_wgt(quad);
    u = FFT_antiderivative(dudt.*dx_wgt,elm);
end