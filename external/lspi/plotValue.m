function []=plotValue(policy)
figure;
S = convertWS(policy.weights);
[X,Y] = meshgrid(-1:0.1:1,-1:0.1:1);
        for i=1:size(X,1)        
            for j=1:size(Y,1)            
                normState=[X(i,j);Y(i,j)];
                Z(i,j)=policy_function(policy, normState); % Z is the action
                Q(i,j)=[normState;Z(i,j)]'*S*[normState;Z(i,j)];  
                Z0(i,j)=0;
                Q0(i,j)=[normState;Z0(i,j)]'*S*[normState;Z0(i,j)];
            end
        end
        colormap(parula);
        %surf(X,Y,Q0,'FaceAlpha',0.5);
        hold on;
        surf(X,Y,Q0);
        colorbar;
        xlabel('Angle')
        ylabel('Velocity')
        zlabel('Qvalue')
        view(3);
        axis square;        
       
        uiwait(gcf); % Wait for the figure to close before proceeding
    
    

end

