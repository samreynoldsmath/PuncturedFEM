function elm = BuildElement(args,quad)
    %{
        asdf
    %}
    switch args.id
        case 'circle'            
            args.num_outer_edges = args.num_edges;
            args.num_inner_edges = 0;
            args.r_out = args.r;
            args.r_in = 0;
            elm = Annulus(args,quad);
        case 'perturbed-circle'
            elm = PerturbedCircle(args,quad);
        case 'teardrop'
            elm = TearDrop(args,quad);
        case 'square'
            args.num_inner_edges = 0;
            elm = SquareCircularHole(args,quad);
        case 'n-gon'
            elm = n_gon(args,quad);
        case 'annulus'
            elm = Annulus(args,quad);
        case 'square-circular-hole'
            elm = SquareCircularHole(args,quad);
        case 'square-square-hole'
            elm = SquareSquareHole(args,quad);
        case 'pegboard'
            elm = Pegboard(args,quad);
        otherwise
            error('Element id "%s" not recognized',args.id)
    end
end

%% Basic building blocks
function [x_fun,dx_fun,ddx_fun] = PolygonFun(x,y)
    m = numel(x);
    if numel(y) ~= m
        error("x and y must be vectors of the same length")
    end
    x_fun = cell(m,1);
    dx_fun = cell(m,1);
    ddx_fun = cell(m,1);
    for edge = 1:m
        edgepp = mod(edge,m)+1;
        x_fun{edge} = @(t) [...
            (0.5*t/pi)*(x(edgepp)-x(edge))+x(edge),...
            (0.5*t/pi)*(y(edgepp)-y(edge))+y(edge)];
        dx_fun{edge} = @(t) [...
            (0.5*ones(size(t))/pi)*(x(edgepp)-x(edge)),...
            (0.5*ones(size(t))/pi)*(y(edgepp)-y(edge))];
        ddx_fun{edge} = @(t) [zeros(size(t)),zeros(size(t))];
    end
end

function [x_fun,dx_fun,ddx_fun] = RegularPolygonFun(num_edges,scale,orient)
    th = orient*(2*pi/num_edges)*(0:num_edges-1);
    x = scale*cos(th);
    y = scale*sin(th);
    [x_fun,dx_fun,ddx_fun] = PolygonFun(x,y);
end

function [x_fun,dx_fun,ddx_fun] = EllipseFun(num_edges,x_radius,y_radius,...
    center,orient)
    x_fun = cell(1,num_edges);
    dx_fun = cell(1,num_edges);
    ddx_fun = cell(1,num_edges);
    for edge = 1:num_edges
        x_fun{edge} = @(t) [...
            x_radius*cos((t+2*pi*(edge-1))/num_edges)+center(1),...
            orient*y_radius*sin((t+2*pi*(edge-1))/num_edges)+center(2)...
            ];
        dx_fun{edge} = @(t) [...
            -x_radius*sin((t+2*pi*(edge-1))/num_edges)/num_edges,...
            orient*y_radius*cos((t+2*pi*(edge-1))/num_edges)/num_edges...
            ];
        ddx_fun{edge} = @(t) [...
            -x_radius*cos((t+2*pi*(edge-1))/num_edges)/(num_edges*num_edges),...
            -orient*y_radius*sin((t+2*pi*(edge-1))/num_edges)/(num_edges*num_edges)...
            ];
    end
end

function [x_fun,dx_fun,ddx_fun] = TearDropKressFun(center,a,b,c,scale)

    x_fun = {@(t) [scale*a*sin(b*t)+center(1),-scale*sin(c*t)+center(2)]};
    dx_fun = {@(t) [scale*a*b*cos(b*t),-scale*c*cos(c*t)]};
    ddx_fun = {@(t) [-scale*a*b*b*sin(b*t),scale*c*c*sin(c*t)]};
end

function [x_fun,dx_fun,ddx_fun] = TearDropFun(center,alpha,scale)

    x_fun = {@(t) [scale*a*sin(b*t)+center(1),-scale*sin(c*t)+center(2)]};
    dx_fun = {@(t) [scale*a*b*cos(b*t),-scale*c*cos(c*t)]};
    ddx_fun = {@(t) [-scale*a*b*b*sin(b*t),scale*c*c*sin(c*t)]};
end

function [x_fun,dx_fun,ddx_fun] = PerturbedCirlceFun(a,b)
    r = @(t) b+a*(1-t/pi).^8;
    dr = @(t) (-8*a/pi)*(1-t/pi).^7;
    ddr = @(t) (56*a/(pi*pi))*(1-t/pi).^6;
    x_fun = {@(t) [...
        r(t).*cos(t), ...
        r(t).*sin(t)]};
    dx_fun = {@(t) [...
        dr(t).*cos(t)-r(t).*sin(t),...
        dr(t).*sin(t)+r(t).*cos(t)]};
    ddx_fun = {@(t) [...
        ddr(t).*cos(t)-2*dr(t).*sin(t)-r(t).*cos(t),...
        ddr(t).*sin(t)+2*dr(t).*cos(t)-r(t).*sin(t)]};
end

%% Simply-connected domains
function elm = PerturbedCircle(args,quad)
    a = args.a;
    b = args.b;
    
    [args.x_fun,args.dx_fun,args.ddx_fun] =...
        PerturbedCirlceFun(a,b);
    
    args.edge_quad_type = 2;
    args.seg_int_pt = [];
    
    args.edge_tag = 0;
    
    elm = class_element;
    elm = elm.init(args,quad);
end

function elm = TearDropKress(args,quad)
    
    center = args.center;
    scale = args.scale;
    a = args.a;
    b = args.b;
    c = args.c;
    
    [args.x_fun,args.dx_fun,args.ddx_fun] = TearDropFun(center,a,b,c,scale);
    
    args.edge_quad_type = 2;
    args.seg_int_pt = [];
    
    args.edge_tag = 0;
    
    elm = class_element;
    elm = elm.init(args,quad);
end

% function elm = TearDropKress(args,quad)
%     
%     center = args.center;
%     scale = args.scale;
%     alpha = args.alpha;
%     
%     [args.x_fun,args.dx_fun,args.ddx_fun] = TearDropFun(center,a,b,c,scale);
%     
%     args.edge_quad_type = 2;
%     args.seg_int_pt = [];
%     
%     elm = class_element;
%     elm = elm.init(args,quad);
% end

function elm = n_gon(args,quad)
    [args.x_fun,args.dx_fun,args.ddx_fun] ...
        = RegularPolygonFun(args.num_edges,args.scale,1);
    
    args.edge_quad_type = 2*ones(1,args.num_edges);
    args.seg_int_pt = [];
    
    args.edge_tag = zeros(1,args.num_edges);
    
    elm = class_element;
    elm = elm.init(args,quad);
end

%% Multiply-connected domains
function elm = Annulus(args,quad)

    if isfield(args,'edge_quad_type')
        if numel(args.edge_quad_type) == args.num_edges
            args.edge_quad_type = args.edge_quad_type;
        else
            warning('quad_type set to default (trapezoid)')
            args.edge_quad_type = ones(1,args.num_edges);
        end
    else
        warning('quad_type set to default (trapezoid)')
        args.edge_quad_type = ones(1,args.num_edges);
    end

    num_outer_edges = args.num_outer_edges;
    num_inner_edges = args.num_inner_edges;
    num_edges = num_outer_edges + num_inner_edges;
    
    args.x_fun = cell(num_edges,1);
    args.dx_fun = cell(num_edges,1);
    args.ddx_fun = cell(num_edges,1);
    
    %% outer boundary
    x_radius = args.r_out;
    y_radius = args.r_out;
    center = [0,0];
    orient = 1;
    [x,dx,ddx] = EllipseFun(num_outer_edges,x_radius,y_radius,center,orient);
    
    for i = 1:num_outer_edges
        args.x_fun{i} = x{i};
        args.dx_fun{i} = dx{i};
        args.ddx_fun{i} = ddx{i};
    end
 
    %% inner boundary
    if num_inner_edges > 0
        
        x_radius = args.r_in;
        y_radius = args.r_in;
        center = [0,0];
        orient = -1;
        [x,dx,ddx] = EllipseFun(...
            num_inner_edges,x_radius,y_radius,center,orient);

        for i = 1:num_inner_edges
            ii = i+num_outer_edges;
            args.x_fun{ii} = x{i};
            args.dx_fun{ii} = dx{i};
            args.ddx_fun{ii} = ddx{i};
        end

        args.seg_int_pt = center;
    
    else
        args.seg_int_pt = [];
    end
    
    %%
    args.edge_tag = zeros(1,num_edges);
    
    %%
    elm = class_element;
    elm = elm.init(args,quad);

end

function elm = SquareCircularHole(args,quad)

    if ~isfield(args,'num_inner_edges')
        args.num_inner_edges = 1;
    end
    if ~isfield(args,'r_in')
        args.r_in = 0.25;
    end

    args.num_edges = 4+args.num_inner_edges;

    % unit square with circular hole of radius r
    args.edge_quad_type = [2*ones(4,1);ones(args.num_inner_edges,1)];
    
    args.x_fun = cell(args.num_edges,1);
    args.dx_fun = cell(args.num_edges,1);
    args.ddx_fun = cell(args.num_edges,1);
    
    %% outer boundary 
    x = [0,1,1,0];
    y = [0,0,1,1];
    [x,dx,ddx] = PolygonFun(x,y);
    
    for i = 1:4
        args.x_fun{i} = x{i};
        args.dx_fun{i} = dx{i};
        args.ddx_fun{i} = ddx{i};
    end
    
    %% inner boundary
    if args.num_inner_edges > 0
        
        x_radius = args.r_in;
        y_radius = args.r_in;
        center = [0.5,0.5];
        orient = -1;
        [x,dx,ddx] = EllipseFun(...
            args.num_inner_edges,x_radius,y_radius,center,orient);

        for i = 1:args.num_inner_edges
            args.x_fun{i+4} = x{i};
            args.dx_fun{i+4} = dx{i};
            args.ddx_fun{i+4} = ddx{i};
        end

        args.seg_int_pt = center;
    
    else
        args.seg_int_pt = [];
    end
    
    %%
    args.edge_tag = [1,2,1,2,3,4];

    %%
    elm = class_element;
    elm = elm.init(args,quad);
    
end

function elm = SquareSquareHole(args,quad)

    num_edges = 8;

    % unit square with circular hole of radius r
    args.edge_quad_type = 2*ones(num_edges,1);
    
    args.x_fun = cell(num_edges,1);
    args.dx_fun = cell(num_edges,1);
    args.ddx_fun = cell(num_edges,1);
    
    %% outer boundary 
    x = [0,1,1,0];
    y = [0,0,1,1];
    [x,dx,ddx] = PolygonFun(x,y);
    
    for i = 1:4
        args.x_fun{i} = x{i};
        args.dx_fun{i} = dx{i};
        args.ddx_fun{i} = ddx{i};
    end
    
    %% inner boundary
    x = args.inner_square_scale*([0,0,1,1]-0.5)+0.5;
    y = args.inner_square_scale*([0,1,1,0]-0.5)+0.5;
    [x,dx,ddx] = PolygonFun(x,y);
    
    for i = 1:4
        args.x_fun{i+4} = x{i};
        args.dx_fun{i+4} = dx{i};
        args.ddx_fun{i+4} = ddx{i};
    end

    %%
    args.seg_int_pt = [0.5,0.5];
    
    elm = class_element;
    elm = elm.init(args,quad);
    
end


% function elm = Pegboard(args,quad)
% 
%     h = 1/args.r;
%     rad_in = 0.25*h;
%     args.num_edges = 4+args.r;
% 
%     % unit square with circular hole of radius r
%     args.edge_quad_type = [2*ones(4,1);ones(args.num_inner_edges,1)];
%     
%     args.x_fun = cell(args.num_edges,1);
%     args.dx_fun = cell(args.num_edges,1);
%     args.ddx_fun = cell(args.num_edges,1);
%     
%     %% outer boundary 
%     x = [0,1,1,0];
%     y = [0,0,1,1];
%     [x,dx,ddx] = PolygonFun(x,y);
%     
%     for e = 1:4
%         args.x_fun{e} = x{e};
%         args.dx_fun{e} = dx{e};
%         args.ddx_fun{e} = ddx{e};
%     end
%     
%     %% inner boundary
%     x_radius = rad_in;
%     y_radius = rad_in;
%     args.seg_int_pt = zeros(args.r,2);
%     num_inner_edges = args.r;
%     for k = 1:num_inner_edges
%         j = 1+mod(k-1,args.r);
%         i = 1+(k-j)/args.r;
%         center = h*([0.5,0.5]+[i-1,j-1]);
%         orient = -1;
%         [x,dx,ddx] = EllipseFun(...
%             num_inner_edges,x_radius,y_radius,center,orient);
% 
%         for e = 1:args.num_inner_edges
%             args.x_fun{e+4+k-1} = x{e};
%             args.dx_fun{e+4+k-1} = dx{e};
%             args.ddx_fun{e+4+k-1} = ddx{e};
%         end
% 
%         args.seg_int_pt(k,:) = center;
%     end
%     
%     %%
%     args.edge_tag = [];
% 
%     %%
%     elm = class_element;
%     elm = elm.init(args,quad);
%     
% end