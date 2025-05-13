%%
% [t,xp] = ode*(@(t,x) thisode(t,x,u), time, x);
function xp = invertPendulum(t,x,u)
g = 9.8;    %gravity constant
m = 2;      % mass of the pendulum
M = 8;      % mass of the cart 
l = 0.5;    % 2l is the length of the pendulum
            % u is the force applied to the cart(in Newtons)
alpha = 1/(m+M);

numerator = g*sin(x(1))-alpha*m*l*x(2)^2*sin(2*x(1))/2-alpha*cos(x(1))*u;
denominator = 4*l/3 - alpha*m*l*cos(x(1))^2; 
xp = [x(2); numerator/denominator];
end