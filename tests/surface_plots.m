%% Surface Plots

% MATLAB syntax:
%   x*y is x times y
%   x/y is x divided by y
%   x^y is x to the y
%   sqrt(x) is the square root of x
%   exp(x) is e to the x
%   log(x) is natural logarithm of x
%   sin(x), cos(x), tan(x), ... are trig functions
%   asin(x), acos(x), atan(x), ... are inverse trig functions
%   pi = 3.14159...

%% z = f(x,y)
% z=@(x,y) 0.5*(x^2+y^2)*log(x^2+y^2)-(x^2+y^2)+x*y*(atan(abs(x/y))+atan(abs(y/x)));
% z=@(x,y) x*log(x^2+y^2)-x+y*(atan(abs(x/y))+atan(abs(y/x)));
% z = @(x,y) (exp(x).*cos(y)-1)./( 1+exp(2*x)-2*exp(x).*cos(y) );
% z = @(x,y) x*y/(x^2+y^2);
z = @(x,y) atan(abs(y/x));

%% Rectangular domain: xmin < x <xmax, ymin < y < ymax
xmin = -0.456;
xmax = 0.456;
ymin = -0.456;
ymax = 0.456;

%% Make mesh, get z values
n=50;
hx=(xmax-xmin)/n;
hy=(ymax-ymin)/n;

x=xmin:hx:xmax;
y=ymin:hy:ymax;
[X,Y]=meshgrid(x,y);

Z=zeros(n+1,n+1);

for i=1:n+1
    for j=1:n+1
        Z(i,j)=z(X(i,j),Y(i,j));
    end
end

%% Make plots
figure(1),clf

subplot(1,2,1) % surface plot
mesh(X,Y,Z)
grid on
axis square
xlabel('x')
ylabel('y')
zlabel('z')

subplot(1,2,2) % contour plot
contour(X,Y,Z,50)
grid on
axis square
xlabel('x')
ylabel('y')
colorbar
