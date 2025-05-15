% MATLAB script in Labview
% input:
%   TrainedADP -- well learned mat for mADP extraction (string input)
%   initialstep -- tuning cycle counter (integer input)
%   IPold -- old impedance parameters (matrix input)
%   kneeData -- knee kinematic data(matrix input)
%   SP - save path (path input)
%%
global mPI;
global mBO;
global mInterface;
global InitImpedance;
global failureFlag;

% gcp; % parallel computing (moved it treadmill starting)

normalValue=[8 8 8 8; 0.24 0.24 0.24 0.24]'; % associate with mInterface parameter
hsOffBound=3;
actionbound=0.5*[  0.01     0.01       0.1        0.01;
                2        3        6         2;
              0.001   0.001     0.001     0.001];
SP = [SP,'mat'];
mkdir(SP);
safetyReset=zeros(4,1); % safetyReset flag
noiseRatio = [0.5, 0.8, 0.5, 0.5];
consecutiveNum=5; % number of samples to be checked for telling success
% majorVote=8; % majority vote threshold
% PI_errorTol=[1.5;0.24]; % 1 degree for peak error, 3% for duration error (alternatively decide whether converged)
majorVote=4; % majority vote threshold
PI_errorTol=[2;0.24]; % 1 degree for peak error, 3% for duration error (alternatively decide whether converged)
Switch_errTol=[0.0527,0.6]; % 0.0527=(1.5/8)^2+0.5*(0.045/0.24)^2
badInitial=0;

%% BO settings
% stopping condition 1. max 15-20? 2. diff in hyper<TH 
% 3. diff in mBO.min_prediction<10%MAX (0.2?)
PassiveKnee=0; % 22<=>6, 25<=>8
is_abs_SI=true;
delta_feature=[4,4,4,4];
feature_lbounds=[4;-5;44;0;];
feature_ubounds=[16;14;58;3];
num_psamples=6;
BO_freq=1;
num_opt_queries=5;
lookback_window_for_sampling=5;
window_threshold=3;
sampling_threshold=2;
% evaluate_flag->0: BO; 
% evaluate_flag->1: fixed impedance evaluation
% evaluate_flag->2: fixed target evaluation (set in Labview)
evaluate_flag=0; 

%%
try
%% Initilization   
    featureT = 0;

    if initialstep == 0
        % create mInterface object
        failureFlag = 0;  %%%%%% TBD          
        impedance = IPold(1:3,[1,2,4,5])';  
        InitImpedance = impedance;
        mInterface = Interface_PI();
        mInterface.normalRange=normalValue;
        mInterface.normalAction=actionbound';
        mInterface.goniometerBias = 0; % addjust according to calibrate
        % featureTarget=init_featureTarget;
        % fixed_featureTarget = [8    0.17    1    0.25   50    0.32    1   0.26]; % original
        fixed_featureTarget = [7    0.2    0    0.3   45    0.26    1    0.24];
        featureTarget = fixed_featureTarget;
        mInterface = mInterface.initHP(impedance, kneeData, fixed_featureTarget);  
        
        if ~isempty(TrainedADP)
            %clear mPI;
            load(TrainedADP, 'mPI');
            % force to turn on policy update
            % mPI.OnlineMode = 1;
        else
            % create Policy iteration object
            mPI = LQR_PolicyIteration(3, 3, 4); % number of state, number of action, number of phase
            mPI.normalState=normalValue; 
            mPI.actionbound=actionbound; 
            mPI.epsilon_zero=0.6;
            mPI.epsilon_d=0.7;
            mPI.OnlineMode=1;     
            for phase=1:4
                 mPI.blocks{phase}.S=full(sprandsym(mPI.nSA,1,[1 2 2 5 1 1]));
                 % mPI.blocks{phase}.S=zeros(mPI.nSA,mPI.nSA);
            end
        end 
        mBO = BayesianOpt(4,'EI',false,is_abs_SI,(feature_lbounds+feature_ubounds)/2,(feature_ubounds-feature_lbounds)/2,feature_lbounds,feature_ubounds);
        mBO.num_psamples=num_psamples;
        
        if ~evaluate_flag
            mBO.initial_samples=[gibbsSampler(1,[-4;7;5;-1;-44;47;0;3],[5;-2;45;1]);          
                                 gibbsSampler(1,[-13;16;5;-1;-44;47;0;3],[15;-2;45;1]);                     
%                                  gibbsSampler(1,[-13;16;-1;5;-44;47;0;3],[15;3;45;1]);     
                                 gibbsSampler(1,[-13;16;-1;5;-55;58;0;3],[15;3;56;1]); 
%                                  gibbsSampler(1,[-13;16;-6;9;-55;58;0;3],[15;8;56;1]);   
                                 gibbsSampler(1,[-13;16;-6;9;-44;47;0;3],[15;8;45;1]);
                                 gibbsSampler(1,[-13;16;-12;14;-55;58;0;3],[15;13;56;1]);  
                                 gibbsSampler(1,[-4;7;-1;5;-55;58;0;3],[5;3;56;1]);]';                           
            % 1-6-3-2-4-5 (ML), 3-2-4-5-6-1, 4-5-3-2-1-6(MH), 2-3-4-5-6-1 (BD)                
            mBO.initial_samples=mBO.initial_samples(:,[2,3,4,5,6,1]);
            
            featureTarget(1:2:8)=mBO.initial_samples(:,1)';
            init_featureTarget=featureTarget;
        end
        save([SP, '\DataFlagRI.mat'],'kneeData','mPI','mInterface');
        
    elseif initialstep==resumeBaseNum
        load(TrainedADP, 'mPI', 'mInterface', 'InitImpedance', 'mBO', 'fixed_featureTarget');
        save([SP, '\DataFlagRI.mat'],'kneeData','mPI','mInterface');
        
    else
        mInterface = mInterface.updateTarget(fixed_featureTarget);
    end       
    save([SP, '\DataB',num2str(initialstep),'.mat' ]);
%% Process Data
    % reinit evaluation threshold after failure happens (TBD) 
    if failureFlag
        failureFlag = 0;
        mInterface = mInterface.failureReinit(kneeData);
    end

    % get state(normalized) and convert state to colunm vectors (2x4)
    kneeDataF=kneeData;
    kneeDataF(:,1)=filtfilt(mInterface.Filter,1,kneeData(:,1));
    kneeDataF(:,3)=filtfilt(mInterface.Filter,1,kneeData(:,3));
    % ------- process hip angle from goniometer -------
    kneeDataF(:,13) = mInterface.goniometerAngleConverter(kneeData(:,13), mInterface.goniometerBias);
    %---------------------------------------------------------
    [mInterface, featureT,hsOffsetMean] = mInterface.evaluate(kneeDataF);
    [fullState] = mInterface.getState(); % fullState includes row vectors, each containing 4 elements, for whole phases
    % [mInterface, temporalSI, meanIntactGaitDuration, meanProthesisGaitDuration] = mInterface.getSymmetry(kneeDataF); % add symmetry index as the third row to expand the state space
    % % add the square of SI (SI^2)
    % if isnan(temporalSI)
    %     error('temporalSI calculation error'); 
    % end
    
    % get GRFs
    if PassiveKnee<1
        % under the current setting, 4 is knee load cell, 6 / 8 is left /
        % right belt. if 4 looks close to 6, then ch6 is prostheis side,
        % otherwise ch 8 is prosthesis side.
        if sum(abs(kneeDataF(:,4)-kneeDataF(:,6)))<sum(abs(kneeDataF(:,4)-kneeDataF(:,8)))
            ps=6;
            is=8;
        else
            is=6;
            ps=8;
        end
        p_GRF=[kneeDataF(:,ps),kneeDataF(:,ps-1),kneeDataF(:,ps)];
        i_GRF=[kneeDataF(:,is),kneeDataF(:,is-1),kneeDataF(:,is)];
        loadcell_GRF=unBiasGRF(repmat(kneeDataF(:,4),1,3));
        p_GRF=unBiasGRF(p_GRF);
        i_GRF=unBiasGRF(i_GRF);
        %     p_GRF=kneeDataF(:,4)+min(min(kneeDataF(:,is),min(kneeDataF(:,ps))))-min(kneeDataF(:,4));
        %     i_GRF=kneeDataF(:,is)-p_GRF+kneeDataF(:,ps);
        p_GRF(:,3)=loadcell_GRF(:,3)+min(min(i_GRF(:,3),min(p_GRF(:,3))))-min(loadcell_GRF(:,3));
        i_GRF(:,3)=i_GRF(:,3)-p_GRF(:,3)+p_GRF(:,1);
    else
        if PassiveKnee==22
            is=25;
        elseif PassiveKnee==25   
            is=22;
        else
           error('Impossible GRF Reading of Passive Knee from Treadmill!'); 
        end
        ps=PassiveKnee;
        
        p_GRF=[kneeDataF(:,ps),kneeDataF(:,ps-1),kneeDataF(:,ps)];
        i_GRF=[kneeDataF(:,is),kneeDataF(:,is-1),kneeDataF(:,is)];  
        p_GRF=unBiasGRF(p_GRF);
        i_GRF=unBiasGRF(i_GRF);
    end
    
    ge=getGaitEvent2(i_GRF,p_GRF);
%     [~,range_info]=getTemporalInfo(ge);
%     stance_time=[mean(range_info.Stance_RangeP(:,2)-range_info.Stance_RangeP(:,1)),...
%                 mean(range_info.Stance_RangeI(:,2)-range_info.Stance_RangeI(:,1))];
%     st_index=2*(stance_time(2)-stance_time(1))/(stance_time(2)+stance_time(1));
    [apimpulse,verticalimpulse,mlimpulse]=getImpulse(p_GRF, i_GRF, ge);
    pimpulse_index=mean(2*(apimpulse(:,2)-apimpulse(:,5))./(apimpulse(:,2)+apimpulse(:,5)));

    %------ have a section code for hip feature extraction----- and modify
    %state. Also modify the costState, need to figure out a way to
    %normalize the hip angle difference
    hip_feature_target = -13;
    hip_vf_feature_target = -9;
    vf_on = 0;
    hip_feature_norm = 20;
    [hip_feature, hip_pkErr, hip_min_angle] = getHipFeature(kneeDataF, hip_feature_target, hip_vf_feature_target, hip_feature_norm, vf_on);
    %------------------------------------------------------------------------
    
    state=[fullState(:,1:2)'; [hip_feature / 50, hip_feature, hip_feature / 50, hip_feature / 50]]; % Phase 1~4 has separate settings
    savedState=state;
    
    costState = state;
    % costState(1, 1) = costState(1, 1) + 0.015;
    % pimpulse_index normalization
    normalized_pimpulse_index = normalizePiImpulse(pimpulse_index);
    % hip joint angle feature normalization
    hip_feature_normalized = hip_feature;
    % cut angle and duration with a level
    angle_bound = 0.25;
    time_bound = 0.085;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % if enabled, the error smalled than a defined bound will be
    % overwrite as a value close to zero, then saved into the costState
    % for i = 1 : 4
    %     % angle
    %     if abs(costState(1, i)) < angle_bound
    %         % costState(1, i) = 0.001; % did not put as 0 to avoid calculation issue
    %     end
    %     % timing
    %     if abs(costState(2, i)) < time_bound
    %         % costState(2, i) = 0.001;
    %     end
    %     % overwrite with normalzied impulseSI
    %     costState(3, i) = state(3, i);
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%% Bayesian Optimization   
%     savedState(2,:)=0; % ignore duration
    fea_measured=mean(featureT);
    fea_error=fea_measured(1:2:8)-featureTarget(1:2:8);
    fea_norm=(fea_measured(1:2:8)'-mBO.mean_input)./mBO.std_input;
    mBO.reset_log=[mBO.reset_log;0];
    if (mBO.quasi_steady==0&&max(abs(fea_error))>sampling_threshold) || evaluate_flag
        mBO=mBO.addSamples(fea_norm.*mBO.std_input+mBO.mean_input,featureTarget(1:2:8)',pimpulse_index,mBO.quasi_steady);        
    else
        mBO.quasi_steady=mBO.quasi_steady+1;
        mBO=mBO.addSamples(fea_norm.*mBO.std_input+mBO.mean_input,featureTarget(1:2:8)',pimpulse_index,mBO.quasi_steady);        
        %%
        if mBO.quasi_steady==window_threshold
%             [mBO,update_sample_list]=mBO.find_cloest_list(size(mBO.samples,2),lookback_window_for_sampling,window_threshold);
            update_sample_list=size(mBO.samples,2)-mBO.quasi_steady+1:size(mBO.samples,2);
            selected_samples=(mBO.samples(1:4,update_sample_list)-mBO.mean_input)./mBO.std_input;
            selected_responses=mBO.responses(update_sample_list);

            if mBO.buffer_push_cnt>=mBO.num_psamples-1 && mod(mBO.buffer_push_cnt+1-mBO.num_psamples,BO_freq)==0
              mBO=mBO.updateModel(selected_samples,selected_responses,false);
              mBO.buffer_push_cnt=mBO.buffer_push_cnt+1;
              mBO=mBO.findBest();
    %           new_fea_norm=mBO.nextSample(num_opt_queries,fea_measured(1:2:8)',delta_feature');
              new_fea_norm=mBO.nextSample(num_opt_queries);
%               featureTarget(1:2:8)=(new_fea_norm.*mBO.std_input+mBO.mean_input)';
            else             
              mBO=mBO.updateModel(selected_samples,selected_responses,true); 
              mBO.buffer_push_cnt=mBO.buffer_push_cnt+1;
%               featureTarget(1:2:8)=mBO.initial_samples(:,mBO.buffer_push_cnt+1)';        
            end
            if strcmp(mBO.switch_mode,'reset')
                mBO.reset_log(end)=1;
                closest_i=mBO.findClosest(featureTarget(1:2:8),size(mInterface.ihist{1},1));
                mBO.impedance_to_reset=[mInterface.ihist{1}(closest_i,:);
                                        mInterface.ihist{2}(closest_i,:);
                                        mInterface.ihist{3}(closest_i,:);
                                        mInterface.ihist{4}(closest_i,:)];    
            end
            mBO.quasi_steady=0;
        end

    end
%% Safety Check (reset IP parameters)& Add samples
    action=zeros(3,4);
    actionTake=zeros(3,4);
    % prepare for IRL customized cost function
    customCost = zeros(mPI.nState, mPI.nState, mPI.blockNum);
    % Mike
    Q = [1, 0.2, 0.01;
        0.98, 0.05, 0.18;
        1, 0.2, 0.01;
        1, 0.2, 0.01];
    % Brendan
    % Q = [1, 0.2, 0.01;
    %     0.99, 0.01, 0.1;
    %     1, 0.2, 0.01;
    %     1, 0.2, 0.01];
    % Woolim
    % Q = [1, 0.2, 0.01;
    %     0.95, 0.01, 0.15;
    %     1, 0.2, 0.01;
    %     1, 0.2, 0.01];
    % Amirreza
    % Q = [1, 0.2, 0.01;
    %     0.85, 0.05, 0.53;
    %     1, 0.2, 0.01;
    %     1, 0.2, 0.01];
    % Wentao
    % Q = [1, 0.2, 0.01;
    %     0.89, 0.1, 0.4;
    %     1, 0.2, 0.01;
    %     1, 0.2, 0.01];
    %     Q = [    
    %     0.6    0.3    0.3
    %     0.6    0.3    0.3
    %     0.6    0.3    0.3
    %     0.6    0.3    0.3
    %     ];
    for phase = 1:4
        customCost(:, :, phase) = diag(Q(phase, :));
    end
    for phase=1:4
%          absState=mInterface.performanceState(phase,1); % absolute value of peak feature
% 
%          if abs(absState) > mPI.upperbound(phase)
%               if initialstep==0
%                  error(num2str(phase)); 
%               else
%                   disp('Out of safety range! Reset to initial impedance and state...');
%                   % reset to initial IP          
%                   mInterface.impedance(phase,:)=InitImpedance(phase,:);
%                   savedState(:,phase) = mPI.blocks{phase}.stateHist(:,1);            
%                   safetyReset(phase)=1;
%               end
%          elseif phase==4 && hsOffsetMean>hsOffBound
%               if initialstep==0
%                  error('5'); 
%               else
% %                   mInterface.impedance(phase,:)=InitImpedance(phase,:);
% %                   savedState(:,phase) = mPI.blocks{phase}.stateHist(:,1);            
% %                   safetyReset(phase)=1; 
%               end
%          end      
         %%%%%%%% get action (normalized) %%%%%%%% 
         if ~evaluate_flag && mBO.reset_log(end)~=1
            [action(:,phase),actionTake(:,phase)] = getAction(mPI,savedState,phase,0,costState); 
            if (phase == 1 && costState(1, 1) <= 0.21) % if error is within bound, constrain action
                actionTake(:,phase) = actionTake(:,phase) * 0.2;
            end
         elseif ~evaluate_flag
             temp=(mBO.impedance_to_reset-mInterface.impedance)./mInterface.normalAction;
             action(:,phase)=temp(phase,:)';
             actionTake(:,phase)=action(:,phase);
             safetyReset(phase)=1;
         elseif evaluate_flag>1
             % (evaluation for a fixed target set in Labview)
             [action(:,phase),actionTake(:,phase)] = getAction(mPI,savedState,phase,0,costState);
         else
             % pass, do nothing (evaluation for a fixed impedance)
         end
         
        %%%%%% add sample %%%%%%%           
        if phase==mPI.blockNum  
           nextPhase=1;              
        else
           nextPhase=phase+1;
        end

        if isempty(mPI.blocks{phase}.stateHist)
           nextState=[];
        else
           nextState= state(:,phase);
        end
        mPI = addSamplePi(mPI,savedState(:,phase),nextState, actionTake(:,phase),action(:,phase),state(:,nextPhase),safetyReset(phase),phase,customCost,costState(:,phase));   
    end  
    %%%%%%%% get new impedances %%%%%%%%
    [mInterface, impedance] = mInterface.getImpedance(actionTake',safetyReset);
%% Update the policy
    % update if (1) reset (2) collecting enough sample (3) not converge    
    % mPI.OnlineMode = 1;
    for phase=1:4
        if (mod(mPI.blocks{phase}.CollectNum-1,mPI.blocks{phase}.sampleLength)==0)...
            && (mPI.blocks{phase}.CollectNum~=1)
%             if phase==1&& mPI.blocks{phase}.timeStep<20
%                 mPI.blocks{phase}.active=1;
%             else
            mPI=isSwitch(mPI, mPI.blocks{phase}.timeStep-mPI.blocks{phase}.sampleLength, mPI.blocks{phase}.timeStep-1, phase, Switch_errTol);
%             end
            if (mPI.blocks{phase}.active==1 && mPI.lockPhase(phase)==0) || mPI.OnlineMode==1
                if mPI.blocks{phase}.timeStep<=mPI.windowLength
                    startIdx=1;
                else
                    startIdx=mPI.blocks{phase}.timeStep-mPI.windowLength;
                end
                endIdx=mPI.blocks{phase}.timeStep-1;
                
                [mPI,~]=cvxW(mPI,startIdx,endIdx,phase); % update policy
                mPI.blocks{phase}.updateflag=[mPI.blocks{phase}.updateflag,1]; 
                mPI.blocks{phase}.active=1;
            else
                mPI.blocks{phase}.updateflag=[mPI.blocks{phase}.updateflag,0];
                mPI.blocks{phase}.active=0;
            end
            mPI.blocks{phase}.CollectNum=1;
        else
            mPI.blocks{phase}.updateflag=[mPI.blocks{phase}.updateflag,0];
        end
        
        mPI.blocks{phase}.onFlag=[mPI.blocks{phase}.onFlag,mPI.blocks{phase}.active];
    end    
%%  Check Convergence in Consecutive Samples  
    mPI=isSuccess(mPI,PI_errorTol,consecutiveNum,majorVote,'double');
       
%%  I/O with LabView                 
    mPI = mPI.increaseK();
    statusFlag=mPI.successflag; 
    cvgFlag1=mPI.convergence(1);
    cvgFlag2=mPI.convergence(2);
    cvgFlag3=mPI.convergence(3);
    cvgFlag4=mPI.convergence(4);
    phase1PkErr=mInterface.performanceState(1,1);
    phase2PkErr=mInterface.performanceState(2,1);
    phase3PkErr=mInterface.performanceState(3,1);
    phase4PkErr=mInterface.performanceState(4,1);
    tune1=mPI.blocks{1}.active;
    tune2=mPI.blocks{2}.active;
    tune3=mPI.blocks{3}.active;
    tune4=mPI.blocks{4}.active;
    cost1=mPI.blocks{1}.cost;
    cost2=mPI.blocks{2}.cost;
    cost3=mPI.blocks{3}.cost;
    cost4=mPI.blocks{4}.cost;
    if exist('mBO','var')
        BO_cnt=mBO.buffer_push_cnt;
    else
        BO_cnt=-1;
    end
    
    impedanceT = impedance';
    IPnew = IPold;
    IPnew(1:3,[1,2,4,5]) = impedanceT;
    IPnew(1:3,3) = impedanceT(1:3,3);
    eFlag = 0;
    eMsg = 'No Error';
 catch ME
    eFlag = 1;
    eMsg = ME.message;
    featureT = 0;
    IPnew = IPold;
    if exist('ME','var')==1
        badInitial=str2double(ME.message);        
    end
end
initialstepn = initialstep+1;
save([SP, '\DataA',num2str(initialstep),'.mat' ]);        
