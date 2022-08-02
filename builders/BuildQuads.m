%{
    Constructs three types of quadratures,
    trapezoid, Kress, and Martensen,
    returns them as elements of a 1x3 cell array
%}

function quad = BuildQuads(qargs)
    

    kress = class_quad;
    kress = kress.Kress(qargs.n,qargs.kress_sig);

    trap = class_quad;
    trap = trap.Left(qargs.n);
    
    mart = class_quad;
    mart = mart.Martensen(qargs.n);

    quad = {trap,kress,mart};

end