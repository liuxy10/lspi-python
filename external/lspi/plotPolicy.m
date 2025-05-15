function []=plotPolicy(policy)
figure;
[X,Y] = meshgrid(-1:0.1:1,-1:0.1:1);

for i=1:size(X,1)        
    for j=1:size(Y,1)            
        normState=[X(i,j);Y(i,j)];
        Z(i,j)= policy_function(policy,normState); %mPI.getAction(normState,1,0);            
    end
end

colormap(parula);
surf(X,Y,Z);
xlabel('Angle')
ylabel('Velocity')
zlabel('action')
view(3);
axis square;
uiwait(gcf); % Wait for the figure to close before proceeding
end