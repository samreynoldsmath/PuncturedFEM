%% polygon
xv = [0,4,4,0,0,NaN,1,1,3,3,1];
yv = [0,0,4,4,0,NaN,1,3,3,1,1];

figure(1),clf
plot(xv,yv,'ko-')
axis([-1,5,-1,5])

%% points
x = linspace(-1,5,32);
[xq,yq] = meshgrid(x,x);

%% test
in = inpolygon(xq,yq,xv,yv);

hold on
scatter(xq(in),yq(in),[],'bo','filled')
scatter(xq(~in),yq(~in),[],'ro','filled')
hold off