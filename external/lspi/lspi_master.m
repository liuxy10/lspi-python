clear all

domain = 'HipExo';
maxiterations = 6;
epsilon = 1e-4;
discount = 0.9;
basis = 'basis_quadratic';
algorithm = 2;  %algorithm = ['lsq'; 'lsqfast'; 'lsqbe'; 'lsqbefast'];
initial_data=zeros(24,30000);
training_data=zeros(24,100000);
final_data=zeros(24,30000);
timing_value = [];

%declaring global variables
global stiffness;
global RL_number;
global var_1;
global var_2;

% setting initial values
stiffness=25;
var_1=0.15;
var_2=0.6;

load('init_state.mat');
load('std_state.mat');
load('gait_plot');
load('gait_plot_time');

RL_run_times=15; % RL running times
samples_per_iteration=16;  % samples once time computing
gait_count_total=5;  %  gaits averaged as a sample
stable_times=6;

Qs = diag([5 1]);
Rs = eye(2)*0.001;
stop_RL_reward=0.05;
%stop_RL_state=0.1;
status_word=4663; % motor_status
RL_convergence_buffer=0;


% Setting Up UCL connection
% uc1= udp('127.0.0.1',100,'LocalPort',6789);
% uc1.InputBufferSize=9600;
% uc1.OutputBufferSize=4800;
% uc1.EnablePortSharing='on';
% % uc1.ByteOrder='littleEndian';
% % uc1.LocalHost='192.168.38.55';
% uc1.Timeout=10;
% fclose(uc1);
% fopen(uc1);


%%% Initialize storage for new samples
empty_result.state = [0;0]; % start value,neg peak value, neg peak position, pos peak value and position
empty_result.action = [0;0];
empty_result.reward = 0.0;
empty_result.nextstate = empty_result.state;
empty_result.absorb = 0;

stop_RL_count=0; % iterations for which RL penalty is within threshold
stop_RL_count_enable=1; % flag for RL
RL_enable=1;% RL work flag
reset_count=0;
total_gait_count=0;
Total_RL_enable_LE=[];
RL_samples_count=[];
gait_trial_samples = [];
totalsamples=[];
RLtotalsamples=[];
absorb=0;
resetSampleCount=[];
Total_sample_action=[];
Total_all_policies=[];
final_policies=[];
sample_action_history=[];
RL_end=0;
sample_data=[];
cycle_data_count = 1;
state_history=[];
reward_history=[];
current_time=[];
samples = repmat(empty_result, 1, samples_per_iteration);
policy = initialize_policy(0, discount, basis);
initial_policy=policy;
state=[0.5;0.3];
% Collecting prior data
time_count_start=0;
time_second = 0; %time in counts
time_before_start=10;
initial_data_count=1;

% fwrite(uc1,[0.001,0,0,var_1,var_2],'single'); % write initial data to the controller

% %% collecting initial data
% while(time_before_start>0)
%     rawdata=fread(uc1,24,'single');
%     time_count_start=time_count_start+1;
%     if(mod(time_count_start,500)==0) %500 Hz, so recording data
%         time_before_start=time_before_start-1;
%         disp(['Algorithm starts in:' num2str(time_before_start)]);
%     end
%     initial_data(:,initial_data_count)=rawdata;
%     initial_data_count=initial_data_count+1;
% end

% fwrite(uc1,[0.001,initial_state(1),initial_state(2),var_1,var_2],'single');
% %%

gait_cycle_count=1;
prev_gait_number=0;

gait_counts=1;
RL_iterations=1;
current_sample_number=1;
gait_data_count=0;


while(RL_iterations<=RL_run_times && stop_RL_count<stable_times)
    if(RL_end==1 && stop_RL_count<stable_times)
        RL_end=0;
    end
    RL_number=RL_iterations;
    while(current_sample_number<=samples_per_iteration && stop_RL_count<stable_times)
        sample_action= policy_function(policy,state)
        var_1=var_1+sample_action(1);
        var_2=var_2+sample_action(2);
        
        current_time=[var_1,var_2]
        timing_value=[timing_value;var_1,var_2];
        sample_action_history=[sample_action_history;sample_action'];
        % fwrite(uc1,[stiffness,1.5*initial_state(1),1.5*initial_state(2),var_1,var_2],'single');
        prev_gait_number=rawdata(11);
        while(rawdata(11)==prev_gait_number)
            rawdata=fread(uc1,24,'single');
            
        end
        prev_gait_number=rawdata(11);
        while(gait_counts<=gait_count_total && stop_RL_count<stable_times)
            time_count_start=time_count_start+1;
            if(mod(time_count_start,500)==0)
                disp(['Algorithm_running time:' num2str(time_second)]);
                time_second=time_second+1;
            end
            % rawdata=fread(uc1,24,'single');
            gait_data_count=gait_data_count+1;
            training_data(:,gait_data_count)=rawdata;
            if(rawdata(11)~=prev_gait_number)
                prev_gait_number=rawdata(11);
                clear complete_gait_cycle;
                complete_gait_cycle=incomplete_gait_cycle;
                clear incomplete_gait_cycle;
                
                [a,b]=max(-1*complete_gait_cycle(4,:));
                state_cycle(gait_counts,1)=-a;
                [a,b]=max(complete_gait_cycle(4,:));
                state_cycle(gait_counts,2)=a;
                gait_data{gait_counts}=complete_gait_cycle;
                disp(['Steps in current sample' num2str(gait_counts)]);
                gait_counts=gait_counts+1;
                
                cycle_data_count=1;
            elseif(rawdata(11)==prev_gait_number)
                incomplete_gait_cycle(:,cycle_data_count)=rawdata;
                cycle_data_count=cycle_data_count+1;
                
            end
            
        end
        gait_counts=1;
        for i=1:length(gait_data)
            
            final_gait(i,1:450)=(gait_data{1,i}(4,1:450));
            final_time(i,1:450)=(gait_data{1,i}(21,1:450));
        end
        gait_plot_trial=mean(final_gait,1);
        gait_plot_trial_time=mean(final_time,1);
        hold off
        plot(gait_plot_time,gait_plot,'b');
        hold on
        
        plot(gait_plot_trial_time,gait_plot_trial,'r');
        drawnow;
        gait_trial_samples = [gait_trial_samples;gait_plot_trial];
        next_state= (2*([(mean(state_cycle(:,1))-initial_state(1));(mean(state_cycle(:,2))-initial_state(2))]));
        %          next_state=0.3*([(mean(state_cycle(:,1))-initial_state(1))/(std_state(1));(mean(state_cycle(:,2))-initial_state(2))/(std_state(2))]);
        %        next_state=10*([(mean(state_cycle(:,1))-initial_state(1));(mean(state_cycle(:,2))-initial_state(2));(mean(state_cycle(:,3))-initial_state(3));(mean(state_cycle(:,4))-initial_state(4));(mean(state_cycle(:,5))-initial_state(5))]);
        absorb=0;
        if(current_sample_number==40)
            absorb=1;
        end
        disp(['**********************************Samples_collected*********************************************: ' ]);
        
        disp(['samples_collected' num2str(current_sample_number)]);
        action =sample_action;
        reward = (state)'* Qs * (state) + action'* Rs * action;
        disp(['Current_sample_penalty:' num2str(reward)]);
        samples(current_sample_number).state = state;
        samples(current_sample_number).action = action;
        samples(current_sample_number).reward = reward;
        samples(current_sample_number).nextstate = next_state;
        samples(current_sample_number).absorb = absorb;
        current_sample_number=current_sample_number+1;
        
        
        if(reward<=stop_RL_reward)     % stop_RL_count accumulates until the preset value
            stop_RL_count=stop_RL_count+1;
            RL_convergence_buffer=0;
        end
        if(reward>stop_RL_reward && stop_RL_count<stable_times)     % stop_RL_count accumulates until the preset value
            
            RL_convergence_buffer=RL_convergence_buffer+1;
        end
        if(RL_convergence_buffer>3)
            stop_RL_count=0;
        end
        
        if (stop_RL_count>=stable_times)%stable_times*sample_count_value
            RL_enable=0;
        end
        disp(['Optimal sample counts:' num2str(stop_RL_count)]);
        state=next_state;
        state_history=[state_history;state'];
        reward_history=[reward_history;reward];
    end
    current_sample_number=1;
    sample_data=[sample_data;samples];
    if (stop_RL_count>=stable_times)
        RL_end=1;
    end
    
    if((RL_enable==1)&&(RL_end==0))
        
        disp('-------------------------------------------------');
        disp('Running RL on collected samples...');
        [final_policy, all_policies] = lspi(domain, algorithm, maxiterations, epsilon, samples, basis, discount, policy);
        Total_all_policies=[Total_all_policies,all_policies];
        final_policies=[final_policies,final_policy];
        policy=final_policy;
    end
    disp(['RL iterations run:' num2str(RL_iterations)]);
    RL_iterations=RL_iterations+1;
    
end



time_before_start=20;
final_data_count=1;
while(time_before_start>0)
    rawdata=fread(uc1,24,'single');
    time_count_start=time_count_start+1;
    if(mod(time_count_start,500)==0) %500 Hz, so recording data
        time_before_start=time_before_start-1;
        disp(['Algorithm ends in:' num2str(time_before_start)]);
    end
    final_data(:,final_data_count)=rawdata;
    final_data_count=final_data_count+1;
end
fwrite(uc1,[0,0,0,0.45,0.85],'single');
