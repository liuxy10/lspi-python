classdef LQR_PolicyIteration
   
%%    
    properties
%         timeStep;      % scalar variable for time step
%         iterationStep; % scalar variable for policy iteration 
%         sampleLength;  %  sample number in batch mode  
%         state;                         % state,column vector 
%         stateHist;
%         action;                        % action,column vector
%         actionHist;   
%         cost;                          % instantaneous cost
%         costHist;
        blockNum; % policy iteration task number i.e. gait phases
        blocks;  
        activeBlock; % current iterating task 
        
        nState;  % state number for each policy iteration block
        nAction; % action number for each policy iteration block
        nSA;     % nState+nAction 
        
        Qs;                            % cost matrix for state
        Rs;                            % cost matrix for action
        QsNext;
        
        gamma;% discounted factor
        
        epsilon_zero;% initial exploration probability
        epsilon_d; %exploration decay rate
        epsilon_k; % current exploration probability
                
        OnlineMode;% Online Mode flag
        outputLimit;
        mode;    % 0 enable sample adding mode, 1 diable sample adding mode
%         firstTime;
        k; % iteration number
        successflag; %  4 phase convergence flag: 1 for success
        convergence; %  convergence flag for each phase
        lockPhase;
        lowerbound;  %  tuning lower bound unit: degree
        intermediatebound;
        upperbound;
        timing;      %  update timing
        actionbound;
        normalAction;
        normalState; % values of being used as normalization raw state
        windowLength;
    end
%%    
    methods
        %% Class constructor
        function obj=LQR_PolicyIteration(nState,nAction,blockNum)
            
            if nargin~=3
                fprintf('illegal input argument number to define a LQR_PolicyIteration variable\n ');
                return;
            end
            
            obj.nState=nState;
            obj.nAction=nAction;
            obj.nSA=nState+nAction;
                        
            Q = [1, 0.5]; %, 0.5];        
            R = 0.00001*[1, 1, 1];
            QNext = [0, 0];%, 0];
            obj.Qs = diag(Q);
            obj.Rs = diag(R);
            obj.QsNext=diag(QNext);
           
            obj.blockNum=blockNum;
            obj.blocks=cell(blockNum,1);
            obj.activeBlock=1;
            
            obj.epsilon_zero=0.6;
            obj.epsilon_d=0.7;
            
            obj.gamma=0.9;
            obj.OnlineMode=0;
            obj.outputLimit = [0.2, 0.2, 0.2, 0.2; 0.5 0.5 0.5 0.5]; %[1, 1, 1, 1];            
            obj.mode = 0;            
%             obj.firstTime = true;
            obj.k = 1;
            obj.successflag=0;
            obj.convergence=zeros(1,4);
            obj.lockPhase=zeros(1,4);
            obj.lowerbound=[1,1,1,1];
            obj.intermediatebound=[3,3,3,3];
            obj.upperbound=[8,8,10,5];
            obj.timing=1e6;
            obj.windowLength=30;
            %obj.normalState=[8 8 8 8; 0.4 0.4 0.4 0.4]';
            obj.normalState=[8 8 8 8; 0.24 0.24 0.24 0.24]';
            obj.actionbound=0.5*[  0.1     0.1       0.1        0.1;
                                     1        1        2         1;
                                 0.001   0.001     0.001     0.001];
%             obj.actionbound=[  0.1     0.1      0.01       0.02;
%                                 1        1        2         0.5;
%                               0.001   0.001     0.001     0.001];
%             obj.normalAction=[0.1; 1 ;0.01];
            
            for i=1:blockNum
                obj.blocks{i}.timeStep = 0;
                obj.blocks{i}.iterationStep = 0; 
                % obj.blocks{i}.sampleLength = obj.nSA * (obj.nSA + 1) / 2; % num of upper triangle
                obj.blocks{i}.sampleLength = 15;
                obj.blocks{i}.grabsamplepointer = 1; 
                obj.blocks{i}.resetHist=[];
                obj.blocks{i}.state = [];  % state,column vector
                obj.blocks{i}.stateHist = [];
                obj.blocks{i}.nextState = [];  % state,column vector
                obj.blocks{i}.nextStateHist = [];
                % obj.blocks{i}.costState = [];  % state after apply tolerance bound for cost calc purpose
                % obj.blocks{i}.costStateHist = [];
                obj.blocks{i}.action = []; % policy action,column vector
                obj.blocks{i}.actionHist = [];
                obj.blocks{i}.actionTake = []; % actual action,column vector
                obj.blocks{i}.actionTakeHist = [];
                obj.blocks{i}.cost = [];   % instantaneous cost, scalar value
                obj.blocks{i}.costHist = [];
                obj.blocks{i}.CollectNum=0;% buffered sample number from last update
                obj.blocks{i}.S=zeros(obj.nSA,obj.nSA);
                obj.blocks{i}.oldS=zeros(obj.nSA,obj.nSA);  
                obj.blocks{i}.SHist=[];
                obj.blocks{i}.Suu=zeros(obj.nAction,obj.nAction);
                obj.blocks{i}.Sxu=zeros(obj.nState,obj.nAction);                
                obj.blocks{i}.updateflag=[]; % record polict update timestep 
                obj.blocks{i}.DOC=cell(200,1); % predefined size
                obj.blocks{i}.onPolNum=0;
                obj.blocks{i}.Cphi_zero=[]; % 
                obj.blocks{i}.dphi_zero=[];
                obj.blocks{i}.active=0;
                obj.blocks{i}.onFlag=[]; % tuning is on
%                 obj.blocks{i}.Ks=zeros(nAction,nState);
%                 obj.blocks{i}.Ks=zeros(nAction,nState);    % state-action feedback gain,  action = Ks * state            
%                 obj.blocks{i}.KsHist=[];
%                 obj.blocks{i}.W=zeros(nchoosek(obj.nSA,2)+obj.nSA,1);
%                 obj.blocks{i}.oldW=zeros(nchoosek(obj.nSA,2)+obj.nSA,1);
            end
        end
        
        %% set batch mode sample number(reset policy before setting new sample length)
        function pi = setSampleLength(pi, newsampleLength)
            if max(size(newsampleLength))>1
                fprintf('illegal setSampleLength input argument\n');
                return;
            end
            index=pi.activeBlock; 
            pi.blocks{index}.sampleLength = newsampleLength;
        end
        
        %% add samples into current & Hist state action variables 
        function pi=addSample(pi,state,nextState,actionTake,action,nextPhaseState,resetflag,index)
            % check input dimension, must be column vector
            if (size(state, 1) ~= pi.nState) || (size(state, 2) ~= 1)
                fprintf(['dimension error: state dimension is [',num2str(size(state, 1)),','...
                      ,num2str(size(state, 2)),']\n']);
                return;
            end
            if (size(actionTake, 1) ~= pi.nAction) || (size(actionTake, 2) ~= 1)                                                                                 
                fprintf(['dimension error: actionTake dimension is [',num2str(size(state, 1)),','...                                                                                   
                       ,num2str(size(actionTake, 2)),']\n']);                                                                                                         
                 return;                                                                                                                                              
            end
            if (size(action, 1) ~= pi.nAction) || (size(action, 2) ~= 1)
                fprintf(['dimension error: actionPolicy dimension is [',num2str(size(action, 1)),','...
                      ,num2str(size(action, 2)),']\n']);
                return;
            end
            
%             index=pi.activeBlock;

            pi.blocks{index}.resetHist=[pi.blocks{index}.resetHist,resetflag];

            pi.blocks{index}.state = state;
            pi.blocks{index}.stateHist = [pi.blocks{index}.stateHist,pi.blocks{index}.state];
            pi.blocks{index}.nextState = nextState;
            pi.blocks{index}.nextStateHist = [pi.blocks{index}.nextStateHist,pi.blocks{index}.nextState];
            % pi.blocks{index}.costState = costState;
            % pi.blocks{index}.costStateHist = [pi.blocks{index}.costStateHist,pi.blocks{index}.costState];
            pi.blocks{index}.actionTake = actionTake;
            pi.blocks{index}.actionTakeHist = [pi.blocks{index}.actionTakeHist,pi.blocks{index}.actionTake];
            
            
            pi.blocks{index}.action = action;                       
            pi.blocks{index}.actionHist = [pi.blocks{index}.actionHist,pi.blocks{index}.action];

%             pi.blocks{index}.state_actionTake= [pi.blocks{index}.stateHist; pi.blocks{index}.actionTakeHist];

%             QQ=pi.Qs;
%             if  index==4
%                 QQ(2,2)=0.1;
%             end
                
%             pi.blocks{index}.cost= state' * pi.Qs * state +nextPhaseState'*pi.QsNext*nextPhaseState...
%                                    +actionTake' * pi.Rs * actionTake;
%             % for setting up a state tolerance delta
%             for istate = 1 : pi.nState
%                if state(istate) < 0.005
%                    state(istate) = 0;
%                end
%             end
%             pi.blocks{index}.cost= costState' * customQ(:, :, index) * costState +nextPhaseState'*pi.QsNext*nextPhaseState...
%                                    +actionTake' * pi.Rs * actionTake;
            % pi.blocks{index}.cost= state' * customQ(:, :, index) * state +nextPhaseState'*pi.QsNext*nextPhaseState...
            %                        +actionTake' * pi.Rs * actionTake;
            pi.blocks{index}.cost=  nextPhaseState'*pi.QsNext*nextPhaseState...
                                   +actionTake' * pi.Rs * actionTake;
            pi.blocks{index}.costHist=[pi.blocks{index}.costHist,pi.blocks{index}.cost];
            
            pi.blocks{index}.CollectNum=pi.blocks{index}.CollectNum+1;
            fprintf(['sample added at time step = ', num2str(pi.blocks{index}.timeStep),' during task ',num2str(index),'\n']);
            pi.blocks{index}.timeStep = pi.blocks{index}.timeStep + 1;
        end      
        %% Reset policy
        function pi=resetPolicy(pi)
            index=pi.activeBlock;
%             pi.blocks{index}.Ks=zeros(pi.nAction,pi.nState); %**** reset to zero ??? TBD ******%

%             pi.blocks{index}.KsHist(end-2:end,:)=[]; % delete the policy failing one-step-check in KsHist
%             pi.blocks{index}.Ks=pi.blocks{index}.KsHist(end-2:end,:); % go back to the last Ks (except first iteration which is random action)            
%             pi.blocks{index}.iterationStep=pi.blocks{index}.iterationStep-1;

               
            pi.blocks{index}.SHist(:,end-pi.nSA+1:end)=[];
            if pi.blocks{index}.iterationStep==1
               pi.blocks{index}.S=zeros(pi.nSA,pi.nSA);
            else
               pi.blocks{index}.S=pi.blocks{index}.SHist(:,end-pi.nSA+1:end);
            end
            pi.blocks{index}.Suu = pi.blocks{index}.S(pi.nState+1 : pi.nState+pi.nAction, pi.nState+1 : pi.nState+pi.nAction);
            pi.blocks{index}.Sxu = pi.blocks{index}.S(1:pi.nState, pi.nState+1 : pi.nState+pi.nAction);
            
            fprintf(['policy reset at time step = ', num2str(pi.blocks{index}.timeStep-1),...
                     ', iteration step = ', num2str(pi.blocks{index}.iterationStep),...
                     ', task = ',num2str(index),'\n']);
          
            % relocate the start sample of the next policy update
%             pi.blocks{index}.grabsamplepointer=pi.blocks{index}.grabsamplepointer-pi.blocks{index}.sampleLength; 
            
%%%           not necessarily minus one under some conditions, put the followed statement in the main script.            
%             pi.blocks{index}.iterationStep=pi.blocks{index}.iterationStep-1;
           
%             pi.mode = 0; % enable sample adding    
        end
        %% Update policy
        function pi=updatePolicy(pi)
            index=pi.activeBlock;
            pi=pi.calcAQ;    
%             pi=pi.calcgainM;
            fprintf(['policy update at time step = ', num2str(pi.blocks{index}.timeStep-1),...
                     ', iteration step = ', num2str(pi.blocks{index}.iterationStep),...
                     ', task = ',num2str(index),'\n']);
            pi.blocks{index}.iterationStep=pi.blocks{index}.iterationStep+1; 
%             pi.mode = 1;  % disable sample adding
        end
        %% transit to next active block
        function pi=updateblock(pi)
            
            pi.activeBlock=pi.activeBlock+1;
            
            if pi.activeBlock>=pi.blockNum+1
                 pi.activeBlock=1;
%                  pi.successflag=1; %**************Discussion****************%
            end            
%             pi.mode = 0; % enable sample adding
        end
        %% 
        function pi=increaseK(pi)
            pi.k=pi.k+1;
        end
        %% record policy update timestep
        function pi=Qupdateflag(pi)
            index=pi.activeBlock;
            pi.blocks{index}.updateflag=[pi.blocks{index}.updateflag,pi.blocks{index}.timeStep-1];
        end
        %% Calculate degree of convergence 
        function [DOC,pi]=calDOC(pi)
            index=pi.activeBlock;
            
            t=find(cellfun(@isempty,pi.blocks{index}.DOC)==1,1);
            
%             pi.blocks{index}.DOC{t}=diag(pi.blocks{index}.state_action'*pi.blocks{index}.oldS*pi.blocks{index}.state_action...
%              -pi.blocks{index}.state_action'*pi.blocks{index}.S*pi.blocks{index}.state_action);
            pi.blocks{index}.DOC{t}=norm(pi.blocks{index}.oldS-pi.blocks{index}.S,'fro');
            DOC=pi.blocks{index}.DOC{t};
        end
        %% 10 consecutive samples converged check
        function [pi]=isSuccess(pi,errorTol,consecutiveNum,majorVote,BS)
            if strcmp(BS,'single')
              bs=1; % only peak error accounts for determining convergence 
            elseif strcmp(BS,'double')
              bs=2; % both peak and duration errors account for determining convergence 
            else 
              bs=1; 
            end             
            for index=1:4
                if size(pi.blocks{index}.stateHist,2)>consecutiveNum
                    sampleChecked=pi.blocks{index}.stateHist(1:2,end-consecutiveNum+1:end)...
                                 .*(pi.normalState(index,:)'*ones(1,consecutiveNum)); % reconstruct original data from normalized

                    if sum(sum(abs(sampleChecked(1:bs,:))<=errorTol(1:bs,:)*ones(1,consecutiveNum))==bs)<majorVote... % using predefined errorTol majority vote
                       %&& sum(mean(abs(sampleChecked(1:bs,:)),2)<=errorTol(1:bs,:))<bs                    
%                         if pi.convergence(index)==1                            
%                              if mean(abs(sampleChecked(1,end-2:end)))>pi.intermediatebound(index)
%                                  pi.convergence(index)=0;
%                              else
%                                  pi.convergence(index)=1;
%                              end                            
%                         else
%                             pi.convergence(index)=0;                                              
%                         end
                        pi.convergence(index)=0;
%                         if pi.lockPhase(index)==1 && mean(abs(sampleChecked(1,:)),2)>=2.5
%                             pi.lockPhase(index)=0;                                                        
%                         end
                    else
                        pi.convergence(index)=1;
%                         pi.lockPhase(index)=1;
                        if min(pi.convergence)==1
                            pi.successflag=1; % success if all phase converged
                            pi.OnlineMode=0; % force to activate offline mode 
                        else
                            pi.successflag=0;
                        end                  
                    end
                end
            end
        end     
        %% convert actual states and action into high-dim quadratic poly states
        function qstate=convert2qstate(pi,sa)

            if (size(sa, 1) ~= pi.nSA) || (size(sa, 2) ~= 1)
                fprintf(['dimension error: state_actionTake dimension is [',num2str(size(sa, 1)),','...
                      ,num2str(size(sa, 2)),']\n']);
                return;
            end

            qMatrix=sa*sa';
            qstate=zeros(nchoosek(pi.nSA,2)+pi.nSA,1);
            index=1;

            for i = 1:length(sa*sa') 
                for j = i:length(sa*sa')
                    if j == i
                        qstate(index) = qMatrix(i,j);
                    else
                        qstate(index) = 2 * qMatrix(i,j);
                    end
                    index = index + 1;
                end
            end
        end

        %% Convert W vector to S matrix
        function [pi,S]=convertW2S(pi,W)            
            if size(W,2)>1
               fprintf('dimension error: W can only be a column vector \n');
            elseif size(W,1)~=(nchoosek(pi.nSA,2)+pi.nSA)
               fprintf('dimension error: W has a wrong row number \n'); 
            end

            bu=triu(ones(pi.nSA),0); 
            bu=bu';
            bu(bu==1)=W;
            bu=bu';

            bl=tril(ones(pi.nSA),0); 
            bl(bl==1)=W;

            S=bu+bl;
            S = S - 0.5*diag(diag(S));     
        end
        %% Convert S matrix to W vector
        function [pi,W]=convertS2W(pi,S)
        %      if ~issymmetric(S)
        %          fprintf('input error: S can only be a symmetric matrix \n');
        %      end
             n=size(S,1);
        %      W=zeros((1+n)*n/2,1);
             for i=1:n         
                 for j=i:n
                     m=(n+2-i+n)*(i-1)/2+j-i+1;
                     W(m,1)=S(i,j);             
                 end         
             end    
        end
        %% on/off swith criteria
        function [pi]=isSwitch(pi,Start,End,index,errTol,varargin)
        % errTol(1):cost tolerence
        % errTol(2):percentage tolerence
        % minslp: minimum slope
            if nargin<6
            minslp=-0.01;
            else
            minslp=varargin{1}; 
            end         
            % sequence of absolute peak errors
             % absPkSequence=abs(pi.blocks{index}.stateHist(1,Start:End)*pi.normalState(index,1));
             
            % sequence of cost
            absCstSequence=abs(pi.blocks{index}.costHist(1,Start:End));
            resetIndice=find(pi.blocks{index}.resetHist(:,Start:End)==1);

            % check increasing/decreasing trend
            if isempty(resetIndice) 
                %absCstSequenceFill=filloutliers_MH(absCstSequence,'nearest','ThresholdFactor',3);
                absCstSequenceFill=medfilt1(absCstSequence,3,'truncate');
                if size(absCstSequenceFill,2)>1
                   absCstSequenceFill=absCstSequenceFill';
                end

                xPdct=(1:length(absCstSequenceFill))';
                fitresult=fit(xPdct,absCstSequenceFill,'poly1'); % linear regression
                ci=confint(fitresult); % coefficient confidence interval for linear regression 
                if sum(ci(:,1)<0)==2 % both slopes should be negative
                   if max(ci(1,1)/ci(2,1),ci(2,1)/ci(1,1))<10 % less an order of magnitude difference (10) b/t max/min slopes                      
                       goodTrend=1;                       
                   elseif min(ci(:,1))<minslp % minimum slope is less than -0.1
                       goodTrend=1; 
                   else
                       goodTrend=0;                        
                   end                 
                else
                   goodTrend=0;
                end 
                
                % check error conditions 
                meanErr=mean(absCstSequence);             
                pct=length(find(absCstSequence<=errTol(1)))/length(absCstSequence);
                
                if meanErr<=errTol(1) || pct>=errTol(2)
                    pi.blocks{index}.active=0;               
                elseif goodTrend==1
                    pi.blocks{index}.active=0;
                else
                    pi.blocks{index}.active=1;
                end
                                
            elseif length(resetIndice)==1
                
                if resetIndice(1)< length(absCstSequence)/3                  
                    pieceCurve=absCstSequence(resetIndice(1):end);
                    %pieceCurveFill=filloutliers_MH(pieceCurve,'nearest','ThresholdFactor',3);
                    pieceCurveFill=medfilt1(pieceCurve,3,'truncate');
                    if size(pieceCurveFill,2)>1
                       pieceCurveFill=pieceCurveFill';
                    end
                    xPdct=(1:length(pieceCurveFill))';
                    fitresult=fit(xPdct,pieceCurveFill,'poly1'); % linear regression
                    ci=confint(fitresult); % coefficient confidence interval for linear regression 
                    if sum(ci(:,1)<0)==2 % both slopes should be negative
                       if max(ci(1,1)/ci(2,1),ci(2,1)/ci(1,1))<10 % less an order of magnitude difference (10) b/t max/min slopes                      
                           goodTrend=1;                       
                       elseif min(ci(:,1))<minslp % minimum slope is less than -0.1
                           goodTrend=1; 
                       else
                           goodTrend=0;                        
                       end                 
                    else
                       goodTrend=0;
                    end                    
                    % check error conditions 
                    meanErr=mean(pieceCurve);             
                    pct=length(find(pieceCurve<=errTol(1)))/length(pieceCurve);
                    if meanErr<=errTol(1) || pct>=errTol(2)
                        pi.blocks{index}.active=0;               
                    elseif goodTrend==1
                        pi.blocks{index}.active=0;
                    else
                        pi.blocks{index}.active=1;
                    end                 
                else
                    pi.blocks{index}.active=1;              
                end                                            
            else
                % if multiple times of resets occured, it gotta switch on
                pi.blocks{index}.active=1;
            end        
        end
        
        %% Calculate approximated Q function, aka. the W vector
        function [pi,error,Cphi,dphi]=cvxW(pi,Start,End,index) % Use N+1 states & actions to calculate
        %     index=pi.activeBlock;

            H=pi.blocks{index}.S(pi.nState+1 : pi.nState+pi.nAction, pi.nState+1 : pi.nState+pi.nAction);
            Xu=pi.blocks{index}.S(1:pi.nState, pi.nState+1 : pi.nState+pi.nAction);
            b=pi.actionbound(:,index)./pi.actionbound(:,index);
        %     b=pi.actionbound(:,index)./pi.normalAction;

%             validIdx=Start-1+find(pi.blocks{index}.resetHist(:,Start:End)==0);
            validIdx=Start:End;
            nextS=pi.blocks{index}.nextStateHist(:,validIdx); 
            pRatio=zeros(length(validIdx),1); % 1 for on-policy samples; 2 for off-policy samples
            for i=1:length(validIdx)
                f= nextS(:,i)'*Xu; % for learning process
                piAction(:,i)=quadprog(H,f',[],[],[],[],-b,b); % quadratic programming
                qstate(:,i) = pi.convert2qstate([pi.blocks{index}.stateHist(:,validIdx(i));
                                                 pi.blocks{index}.actionTakeHist(:,validIdx(i))]);
                qstateNext(:,i)=pi.convert2qstate([pi.blocks{index}.nextStateHist(:,validIdx(i));
                                                   piAction(:,i)]);
                 if norm(piAction(:,i)-pi.blocks{index}.actionTakeHist(:,validIdx(i)+1))<1e-7 
                     pRatio(i)=1;                             
                 elseif norm(pi.blocks{index}.nextStateHist(:,validIdx(i))-pi.blocks{index}.stateHist(:,validIdx(i)+1))>1e-7
                     pRatio(i)=1;
                 else
                     pRatio(i)=2;
                 end                           
            end 
            pRatio=diag(pRatio);
            
            ns=length(validIdx);  
%             [~,lastw]=convertS2W(pi,pi.blocks{index}.S);
%             Cphi=1/ns*(qstate*qstate'-pi.gamma*qstate*pRatio*qstateNext')+0*eye(size(qstate,1));
%             dphi=1/ns*(qstate*pi.blocks{index}.costHist(validIdx(1:end-1))')+0*lastw;

            phiw=zeros(size(qstate,1),1);   
%            [~,phiw]=convertS2W(pi,pi.blocks{index}.S);
            
            j=1;
            error=1;
            stepsize=0.5;
            if ~isempty(pi.blocks{index}.Cphi_zero)&& ns<pi.windowLength 
                Cphi=0.5*pi.blocks{index}.Cphi_zero+0.5*1/ns*(qstate*qstate'-pi.gamma*qstate*pRatio*qstateNext');
                dphi=0.5*pi.blocks{index}.dphi_zero+0.5*1/ns*qstate*pRatio*pi.blocks{index}.costHist(validIdx)';
                while j<=1000 && error>1e-4 %  incremental importance sampling 
                    oPhiw=phiw; 
                    residuePhi=phiw-stepsize*(Cphi*phiw-dphi);
                    [~,phiw,dyError]=proDysktra(pi,residuePhi,100,1e-4);                                                
                    j=j+1;
                    error=norm(oPhiw-phiw);
                    stepsize=1/(j+1);
%                    stepsize=1e-2;                    
                end                  
%                 while j<=ns %&& error>1e-4
%                     oPhiw=phiw; 
%                     Cphi=1/(j+pi.windowLength/2)*(Cphi*(j-1+pi.windowLength/2)+(qstate(:,j)*qstate(:,j)'-pi.gamma*qstate(:,j)*pRatio(j,j)*qstateNext(:,j)'));
%                     dphi=1/(j+pi.windowLength/2)*(dphi*(j-1+pi.windowLength/2)+qstate(:,j)*pRatio(j,j)*pi.blocks{index}.costHist(validIdx(j))');
%                     residuePhi=phiw-stepsize*(Cphi*phiw-dphi);
%                     [~,phiw,dyError]=proDysktra(pi,residuePhi,100,1e-4);                                                
%                     j=j+1;
%                     error=norm(oPhiw-phiw);
%                     stepsize=1/(j+1);
%                 end                              
             else % what is running with the pendulum model
                Cphi=1/ns*(qstate*qstate'-pi.gamma*qstate*pRatio*qstateNext'); % transition matrix error 
                dphi=1/ns*qstate*pRatio*pi.blocks{index}.costHist(validIdx)';
                while j<=1000 && error>1e-4 %(j<=ns for offline training)
                    oPhiw=phiw; 
                    residuePhi=phiw-stepsize*(Cphi*phiw-dphi);
                    [~,phiw,dyError]=proDysktra(pi,residuePhi,100,1e-4);  % alternative projection methods                                             
                    j=j+1;
                    error=norm(oPhiw-phiw);
                    stepsize=1/(j+1);
%                    stepsize=1e-3;
                end
%                 Cphi=0;
%                 dphi=0;
%                 while j<=ns %&& error>1e-4
%                     oPhiw=phiw; 
%                     Cphi=1/j*(Cphi*(j-1)+(qstate(:,j)*qstate(:,j)'-pi.gamma*qstate(:,j)*pRatio(j,j)*qstateNext(:,j)'));
%                     dphi=1/j*(dphi*(j-1)+qstate(:,j)*pRatio(j,j)*pi.blocks{index}.costHist(validIdx(j))');
%                     residuePhi=phiw-stepsize*(Cphi*phiw-dphi);
%                     [~,phiw,dyError]=proDysktra(pi,residuePhi,100,1e-4);                                                
%                     j=j+1;
%                     error=norm(oPhiw-phiw);
%                     stepsize=1/(j+1);
%                 end

%                 Cphi=0;
%                 dphi=0;
% %                [~,phiw]=convertS2W(pi,pi.blocks{index}.S);
%                 while j<=ns                    
%                     oPhiw=phiw; 
%                     Cphi=(qstate(:,j)*qstate(:,j)'-pi.gamma*qstate(:,j)*pRatio(j,j)*qstateNext(:,j)');
%                     dphi=qstate(:,j)*pRatio(j,j)*pi.blocks{index}.costHist(validIdx(j))';
%                     residuePhi=phiw-stepsize*(Cphi*phiw-dphi);
%                     [~,phiw,dyError]=proDysktra(pi,residuePhi,100,1e-4);                                                
%                     error=norm(oPhiw-phiw);  
%                     stepsize=1/(j+1);
%                     j=j+1;
%                 end

            end
            [~,S]=convertW2S(pi,phiw);             
            pi.blocks{index}.S=S;
            pi.blocks{index}.SHist=[pi.blocks{index}.SHist,pi.blocks{index}.S];
            pi.blocks{index}.iterationStep=pi.blocks{index}.iterationStep+1;
%             pi.blocks{index}.Suu = pi.blocks{index}.S(pi.nState+1 : pi.nState+pi.nAction, pi.nState+1 : pi.nState+pi.nAction);
%             pi.blocks{index}.Sxu = pi.blocks{index}.S(1:pi.nState, pi.nState+1 : pi.nState+pi.nAction);          
%             pi.blocks{index}.grabsamplepointer=Start+pi.blocks{index}.sampleLength;
        end 
        %% get action (normalized)
        function [action,actionTake]=getAction(pi,state,index,limitFlag)
            
            if (size(state, 1) ~= pi.nState) || (size(state, 2) ~= pi.blockNum)
                fprintf(['dimension error: state dimension for all tasks is [',num2str(size(state, 1)),','...
                      ,num2str(size(state, 2)),']\n']);
                return;
            end
            
%             index=pi.activeBlock;          
%             action=zeros(pi.nAction,pi.blockNum);
            if limitFlag==0
               b=pi.actionbound(:,index)./pi.actionbound(:,index);% action without output limit 
%                b=pi.actionbound(:,index)./pi.normalAction;              
            else 
               b=pi.outputLimit(limitFlag,index)*pi.actionbound(:,index)./pi.actionbound(:,index);% action with output limit
%                b=pi.outputLimit(limitFlag,index)*pi.actionbound(:,index)./pi.normalAction;  
            end
            
%             pi.epsilon_k=pi.epsilon_zero*pi.epsilon_d^(pi.k*0.01);             

            
             
            options= optimset('display','off');
%                H=pi.blocks{index}.Suu;
%                f=2*state(:,index)'*pi.blocks{index}.Sxu;
            H = pi.blocks{index}.S(pi.nState+1 : pi.nState+pi.nAction, pi.nState+1 : pi.nState+pi.nAction);
            f = state(:,index)'*pi.blocks{index}.S(1:pi.nState, pi.nState+1 : pi.nState+pi.nAction); % for action process
            % f = costState(:,index)'*pi.blocks{index}.S(1:pi.nState, pi.nState+1 : pi.nState+pi.nAction); % for action process # ACTUAL ONE
            if pi.blocks{index}.active==1 
%                 if isempty(find(pi.blocks{index}.updateflag,1,'last'))
%                     sub=size(pi.blocks{index}.stateHist,2)-1;
%                 else
%                     sub=find(pi.blocks{index}.updateflag,1,'last');
%                 end
%                 
%                 decayNum=size(pi.blocks{index}.stateHist,2)-sub;
                decayNum=size(pi.blocks{index}.stateHist,2)-find(pi.blocks{index}.updateflag,1,'last');
                pi.epsilon_k=pi.epsilon_zero*pi.epsilon_d^(decayNum);
                epsilon=binornd(1,pi.epsilon_k); 
                if epsilon==1
                    action =quadprog(H,f,[],[],[],[],-b,b,[],options);
                    actionTake=1*(rand(size(pi.nAction))-0.5).*b;
%                 action=action.*pi.normalAction;
                else
                    action=quadprog(H,f,[],[],[],[],-b,b,[],options);
                    actionTake=action;
                end
            else                  
                action=quadprog(H,f,[],[],[],[],-b,b,[],options);
                actionTake=action;
%                 actionTake = actionTake + 0.2*(rand(size(pi.nAction))-0.5).*b;
%                 action=action.*pi.normalAction;
            end
        end
        %% alternating projection
        function [pi,x,error]=proDysktra(pi,x0,ballR,errTol)           
            error=1;
            j=1;
            I=zeros(length(x0),2);
            oldI=zeros(length(x0),2);
            x=x0;
            while j<500 && error>errTol 
                
                oldX=x;
                if norm(x-I(:,1))>ballR
                    x=ballR*(x-I(:,1))/norm(x-I(:,1));                    
                else
                    x=x-I(:,1);
                end                
                oldI(:,1)=I(:,1);
                I(:,1)=x-(oldX-I(:,1));
                
                oldX=x;
                [~,s]=convertW2S(pi,x-I(:,2));
                [V,D] = eig(s);
                D(D<0)=0;                              
                s=V*D*V';
                [~,x]=convertS2W(pi,s);
                oldI(:,2)=I(:,2);
                I(:,2)=x-(oldX-I(:,2));  
                                
                j=j+1;
                error=norm(oldI-I)^2;                
            end            
        end
        
    end
    
    
end

