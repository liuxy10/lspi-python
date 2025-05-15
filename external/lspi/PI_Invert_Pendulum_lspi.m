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
% mPI=LQR_PolicyIteration(2,1,1);
% actionbound=1;
normalAction=1;
normalState=[pi/2;2*pi]; % this need to be defined for every different system
% Qs=[1 0;0 0.3]; % is this one updating
% Rs=0.01;


basis = 'basis_quadratic';
discount = 0.9;
policy = pendulum_initialize_policy(0, discount, basis);

% S=0.3*ones(3,3);  
%rng(5);
% S = convertW2S(policy.weights);
%S=ones(3,3);
%S=S;
% Suu=S(3,3);
% Sxu=S(1:2,3);
[nextState, reward] = model.getState()

%%
active=1;
updateflag=1;
epsilon_zero=0.5;
epsilon_d=0.5;
samples_per_iteration = 300; % number of samples per iteration
empty_result.state = [0;0]; % start value,neg peak value, neg peak position, pos peak value and position
empty_result.action = [0;0];
empty_result.reward = 0.0;
empty_result.nextstate = empty_result.state;
empty_result.absorb = 0;
samples = repmat(empty_result, 1, samples_per_iteration);

% collect samples
failure=0;
for k=1:100 % for each policy update iteration
    
    i=0;
    % convertWS(policy.weights), policy.weights
    while i<samples_per_iteration % for each trial  
        i=i+1;   
        state=nextState;
        normState=state'./normalState;
              
        [actionT,~]=policy_function(policy,normState); 
        
        [model,actionTake]=model.simulate(actionT); 
        
        [nextState, cost] = model.getState(); %
        % fprintf('Policy weights: %s, actionT: %s\n', mat2str(policy.weights), mat2str(actionT)); % Print policy weights and actionT
        
        reward = cost;
        normNextState=nextState'./normalState; % normalize the state 
        if abs(nextState(1))<pi/2 
            % mPI=addSample(normState, normNextState,actionT,actionP,[0;0],0,1); 
            absorb = 0;
            fprintf('Sample %d: normState = [%f, %f], normNextState = [%f, %f], actionT = %f, reward = %f, absorb = %d\n', ...
                i, normState(1), normState(2), normNextState(1), normNextState(2), actionTake, reward, absorb);
            samples = addSample(samples, normState, normNextState, actionT, reward, absorb);
            updateflag=[updateflag,0]; 
        else
            % mPI=addSample(normState, normNextState,actionT,actionP,[0;0],1,1);
            absorb = 1;
            % fprintf('Sample %d: normState = [%f, %f], normNextState = [%f, %f], actionT = %f, reward = %f, absorb = %d\n', ...
                % i, normState(1), normState(2), normNextState(1), normNextState(2), actionT, reward, absorb);
            
            samples = addSample(samples, normState, normNextState, actionT, reward, absorb);
            samples = filterSample(samples, 2 * samples_per_iteration);
            
            updateflag=[updateflag,1];
            failure=1;
            break;
        end
    end
    k=1+k;  
    
    close all;
    % model.showHist;
    
    if failure  % if failure, update the start and end index of the training data   
%      if k<30
       range=find(updateflag==1);
        if numel(range)<=2
                Start=1;
        else
%            Start=1;
%            Start=range(1)+1;
          Start=range(end-2)+1;
        end
       End=size(updateflag,1)-1;
    %    [mPI,~]=cvxW(mPI,Start,End,1); 
       policy = lspi("pendulum", 6, 10e-4, samples, 'basis_quadratic', 0.9, policy);
    %    plotValue(policy);
    %    plotPolicy(policy);
       failure=0;
    else
         break; 
    end
    
    xi=[1;1].*(rand(2,1)-[0.5;0.5]);
    model=model.reset([0;0], xi);    
    [nextState, reward] = model.getState();
end


model.showHist;
plotValue(policy);
plotPolicy(policy);











%% check fitting error

range=find(resetHist==1);
sIdx=1;
% eIdx=range(end);
eIdx=size(nextStateHist,2);
S=blocks{1, 1}.SHist(:,end-5:end-3) % stack of 3*3 matrix 
sa=[blocks{1, 1}.stateHist(:,sIdx:eIdx);blocks{1, 1}.actionTakeHist(:,sIdx:eIdx)]; % state-action stack from the corresponding policy
parfor i=1:size(blocks{1, 1}.nextStateHist(:,sIdx:eIdx),2) % create the action list from the final policy
            f=2*blocks{1, 1}.nextStateHist(:,i)'*Sxu; % create the cost
            pi_action(:,i)=quadprog(Suu,f',[],[],[],[],-1,1, optimset(options, 'Display', 'off'));                
end
spa=[blocks{1, 1}.nextStateHist(:,sIdx:eIdx);pi_action]; % state-action stack from the final policy
Qs=diag(sa'*S*sa); % Q-value from the training data
Qsp=diag(spa'*S*spa); % Q-value from the final policy
figure;
plot(Qs,blocks{1, 1}.costHist(:,sIdx:eIdx)'+gamma*Qsp,'*'); % plot the cost
hold on;
% plot(0:max([max(Qs),max(blocks{1, 1}.costHist(:,sIdx:eIdx)'+gamma*Qsp)]), 0:max([max(Qs),max(blocks{1, 1}.costHist(:,sIdx:eIdx)'+gamma*Qsp)]));
% xlim([0 max([max(Qs),max(blocks{1, 1}.costHist(:,sIdx:eIdx)'+gamma*Qsp)])]);
% ylim([0 max([max(Qs),max(blocks{1, 1}.costHist(:,sIdx:eIdx)'+gamma*Qsp)])]);
axis square;
%%
close all;
xi=[0.4;-0.4];
model=model.reset([0;0], xi);
model.unoise=10;
[nextState, reward] = getState(model);
active=0;
hist=[];
%epsilon_k=comm.internal.BernoulliBinaryGenerator('ProbabilityOfZero', 0, 'SamplesPerFrame', 1);
for n=1:1000
        state=nextState;
%         if abs(nextState(1))>pi/2
%             break;
%         end
        normState=state'./normalState;
        action=getAction(normState,1,0);
        [model,actionTake]=model.simulate(action);
        [nextState, reward] = model.getState();
        hist=[hist,state'];
end
model.showHist; 

%%
% data=[stateHist(:,sIdx:eIdx);actionTakeHist(:,sIdx:eIdx)];
% obj = gmdistribution.fit(data',1);
% plot3(data(1,:),data(2,:),data(3,:),'*');
% xlabel('P');
% ylabel('V');
% zlabel('T');
%%

plotValue(mPI,0);
plotPolicy(mPI);
hold on;
%  save('offP.mat','mPI','model')



%% redefine policy



function samples = addSample(samples, state, next_state, action, reward, absorb)
    if ~exist('samples', 'var') || isempty(samples)
        samples = struct('state', {}, 'action', {}, 'reward', {}, 'nextstate', {}, 'absorb', {}, 'updateflag', {});
    end
    i = length(samples) + 1;
    samples(i).state = state;
    samples(i).action = action;
    samples(i).reward = reward;
    samples(i).nextstate = next_state;
    samples(i).absorb = absorb;
end

function samples = filterSample(samples, num_samples)
    if ~exist('samples', 'var') || isempty(samples)
        samples = struct('state', {}, 'action', {}, 'reward', {}, 'nextstate', {}, 'absorb', {}, 'updateflag', {});
    end
    samples = samples(max(1, length(samples) - num_samples):length(samples));
end