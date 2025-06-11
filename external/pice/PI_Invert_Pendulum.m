%%
% Update once at a time (no iterations till to convergence)
% On-off policy schemes both work!!!
%% Setting

clear all;
model=Inverted_Pendulum(0.01,1);
xi=[0.4;0.2];
model=model.reset([0;0], xi);
model.rewardtype=1;
model.xnorm=[pi/2;2*pi];


%% create the policy
mPI=LQR_PolicyIteration(2,1,1);
mPI.actionbound=1;
mPI.normalAction=1;
mPI.normalState=[pi/2;2*pi]; % this need to be defined for every different system
mPI.Qs=[1 0;0 0.3];
mPI.Rs=0.01;
% mPI.blocks{1}.S=0.3*ones(3,3);
%rng(5);
S= full(sprandsym(3,1,[10 2 10])); % random initial Policy
%S=ones(3,3);
%mPI.blocks{1}.S=S;
% mPI.blocks{1}.Suu=S(3,3);
% mPI.blocks{1}.Sxu=S(1:2,3);
[nextState, reward] = getState(model);

%%
mPI.blocks{1}.active=1;
mPI.blocks{1}.updateflag=1;
mPI.epsilon_zero=0.5;
mPI.epsilon_d=0.5;

% collect samples
failure=0;
for k=1:100 % for each policy update iteration
    
    i=1;
    while i<=1000 % for each trial    
        state=nextState;
        normState=state'./mPI.normalState;
        [actionP,actionT]=mPI.getAction(normState,1,0);
        [model,actionTake]=model.simulate(actionT); 
        [nextState, reward] = model.getState();
        normNextState=nextState'./mPI.normalState; % normalize the state 
        
        if abs(nextState(1))<pi/2 % if the pendulum is not falling
            % fprintf('Sample %d: normState = [%f, %f], normNextState = [%f, %f], actionT = %f, reward = %f', ...
                % i, normState(1), normState(2), normNextState(1), normNextState(2), actionTake, reward);
            mPI=mPI.addSample(normState, normNextState,actionT,actionP,[0;0],0,1); 
            
            mPI.blocks{1}.updateflag=[mPI.blocks{1}.updateflag,0]; 
        else
            mPI=mPI.addSample(normState, normNextState,actionT,actionP,[0;0],1,1);
            % fprintf('Sample %d: normState = [%f, %f], normNextState = [%f, %f], actionT = %f, reward = %f', ...
                % i, normState(1), normState(2), normNextState(1), normNextState(2), actionTake, reward);
            
            mPI.blocks{1}.updateflag=[mPI.blocks{1}.updateflag,1];
            failure=1;
            break;
        end
        i=i+1; 
        mPI.k=1+mPI.k; 
        
        
    end
    
    close all;
    % model.showHist;
    
    if failure  % if failure, update the start and end index of the training data   
%      if k<30
       range=find(mPI.blocks{1}.resetHist==1);
        if numel(range)<=2
                Start=1;
        else
%            Start=1;
%            Start=range(1)+1;
          Start=range(end-2)+1;
        end
       End=size(mPI.blocks{1}.nextStateHist,2)-1;
       [mPI,~]=cvxW(mPI,Start,End,1); 
        % plotValue(mPI,0);
        % plotPolicy(mPI);
        hold on;
       failure=0;
    else
         break; 
    end
    
     xi=[1;1].*(rand(2,1)-[0.5;0.5]);
     model=model.reset([0;0], xi);    
    [nextState, reward] = getState(model);
end

model.showHist;
% plotValue(mPI,0);
% plotPolicy(mPI);

%% check fitting error

range=find(mPI.blocks{1}.resetHist==1);
sIdx=1; % start index of the training data
% eIdx=range(end);
eIdx=size(mPI.blocks{1}.nextStateHist,2);
S=mPI.blocks{1, 1}.SHist(:,end-5:end-3) % stack of 3*3 matrix 
sa=[mPI.blocks{1, 1}.stateHist(:,sIdx:eIdx);mPI.blocks{1, 1}.actionTakeHist(:,sIdx:eIdx)]; % state-action stack from the corresponding policy
parfor i=1:size(mPI.blocks{1, 1}.nextStateHist(:,sIdx:eIdx),2) % create the action list from the final policy
            f=2*mPI.blocks{1, 1}.nextStateHist(:,i)'*mPI.blocks{1}.Sxu; % create the cost
            pi_action(:,i)=quadprog(mPI.blocks{1}.Suu,f',[],[],[],[],-1,1);                
end
spa=[mPI.blocks{1, 1}.nextStateHist(:,sIdx:eIdx);pi_action]; % state-action stack from the final policy
Qs=diag(sa'*S*sa); % Q-value from the training data
Qsp=diag(spa'*S*spa); % Q-value from the final policy
figure;
scatter(Qs, mPI.blocks{1, 1}.costHist(:,sIdx:eIdx)' + mPI.gamma*Qsp, 36, 1:length(Qs), 'filled'); % color by index
% add diagonal line
hold on;
maxVal = max([max(Qs), max(mPI.blocks{1, 1}.costHist(:,sIdx:eIdx)' + mPI.gamma*Qsp)]); % find the maximum value for the diagonal line
lineVals = linspace(0, maxVal, 100);
plot(lineVals, lineVals, 'b--');
grid on;
title('Cost + \gamma*Q_{s''} vs Q_s');
colorbar;
xlabel('Q_s');
ylabel('Cost + gamma*Q_{s''}');
hold on;
% plot(0:max([max(Qs),max(mPI.blocks{1, 1}.costHist(:,sIdx:eIdx)'+mPI.gamma*Qsp)]), 0:max([max(Qs),max(mPI.blocks{1, 1}.costHist(:,sIdx:eIdx)'+mPI.gamma*Qsp)])); % plot Q values
% xlim([0 max([max(Qs),max(mPI.blocks{1, 1}.costHist(:,sIdx:eIdx)'+mPI.gamma*Qsp)])]);
% ylim([0 max([max(Qs),max(mPI.bloc dks{1, 1}.costHist(:,sIdx:eIdx)'+mPI.gamma*Qsp)])]);
axis square;
%%
% close all;
% xi=[0.4;-0.4];
% model=model.reset([0;0], xi);
% model.unoise=10;
% [nextState, reward] = getState(model);
% mPI.blocks{1}.active=0;
% hist=[];
% %mPI.epsilon_k=comm.internal.BernoulliBinaryGenerator('ProbabilityOfZero', 0, 'SamplesPerFrame', 1);
% for n=1:1000
%         state=nextState;
% %         if abs(nextState(1))>pi/2
% %             break;
% %         end
%         normState=state'./mPI.normalState;
%         action=mPI.getAction(normState,1,0);
%         [model,actionTake]=model.simulate(action); 
%         [nextState, reward] = model.getState();
%         hist=[hist,state'];
% end
% model.showHist; 

% %%
% % data=[mPI.blocks{1}.stateHist(:,sIdx:eIdx);mPI.blocks{1}.actionTakeHist(:,sIdx:eIdx)];
% % obj = gmdistribution.fit(data',1);
% % plot3(data(1,:),data(2,:),data(3,:),'*');
% % xlabel('P');
% % ylabel('V');
% % zlabel('T');
% %%

% plotValue(mPI,0);
% plotPolicy(mPI);
% hold on;
% %  save('offP.mat','mPI','model')

% plot(mPI.blocks{1}.stateHist(1,:),mPI.blocks{1}.stateHist(2,:),'r*');
%%
function []=plotPolicy(mPI)
figure;
[X,Y] = meshgrid(-1:0.1:1,-1:0.1:1);

    for i=1:size(X,1)        
        for j=1:size(Y,1)            
            normState=[X(i,j);Y(i,j)];
            Z(i,j)=mPI.getAction(normState,1,0);            
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
function []=plotValue(mPI,offset)
figure;
S=mPI.blocks{1}.SHist(:,end-offset*3-2:end-offset*3);
[X,Y] = meshgrid(-1:0.1:1,-1:0.1:1);
        for i=1:size(X,1)        
            for j=1:size(Y,1)            
                normState=[X(i,j);Y(i,j)];
                Z(i,j)=mPI.getAction(normState,1,0); % Z is the action
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




















