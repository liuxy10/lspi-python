%% Inverted_Pendulum platform
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%
%
classdef Inverted_Pendulum
    properties
        dt;
        
        %% state varibles
        xi;
        x;
        xnorm
        xname;
        
        %% control varibles           
        ui;        
        u;
        ulist;
        utype;
        unoise;      
        uMag;
        
        %% reward variables        
        desire;         % desire value
        sindex;         % target state index
        splus;          % safe range
        sminus;
        rewardtype;     % rewardtype: 1 for smooth reward, 0 for not smooth(1 or 0.1) reward
        costconstant;
        costfailure;
        costsucceed;
        w;
        
        %% history varibles
        uhist;
        xhist;
        histcnt;
        savecnt;
    end
    methods
        function obj = Inverted_Pendulum(uNoise, actiontype, desire)
            %% set default value
            if nargin == 0              
               uNoise = 0;
               desire = [0, 0]';
               actiontype = 1;              % default control type is discrete action
            elseif nargin == 1
               desire = [0, 0]';
               actiontype = 1;              % default control type is discrete action
            elseif nargin == 2
               desire = [0, 0]';                
            end
            
            obj.dt = 0.01;                   % cycle length
            
            %% state Initialization
            obj.xi = [0,0]';                % initial state
            obj.x = obj.xi;
            obj.xnorm = [1; 1];             % normalization of state x
            obj.xname = {'angle', 'anglular velocity'}; % name of the state
            
            %% control Initialization
            obj.ui = 0;                     % initial control(scaler or colum vector)
            obj.u = obj.ui;
            % action type 1 for discrete action, 0 for continous action
            if actiontype == 1
                obj.ulist = [-1; 0; 1];     % action range or action list;
                obj.utype = 3;              % 0 for continue action space; action number for discrete action space
                obj.uMag = 100;              % action magnitude
            else
                obj.ulist = [-1; 0; 1];     % action range or action list;
                obj.utype = 0;              % 0 for continue action space; action number for discrete action space
                obj.uMag = 100;              % action magnitude
            end
            obj.unoise = obj.uMag*uNoise;   % control noise, default is 0
            
            %% reward definition
            obj.rewardtype = 0;             % rewardtype: 1 for smooth reward, 0 for discrete(1 or 0.1) reward
            obj.costsucceed = 0;            % cost for succeed range ( abs(e) < Splus )
            obj.splus = 0;                  % succeed range define
            obj.costconstant = 0;           % cost for normal range( Splus < abs(e) < Sminus )
            obj.costfailure = 1;            % cost for failure range ( abs(e) > Sminus )
            obj.sminus = pi/2;              % failure range define

            obj.desire = desire;            % state target
            obj.sindex = 1;                 % index of the target in the state
            obj.w = atanh(sqrt(0.95))./obj.splus;   %if rewardtype is one, uncomment this line.

            %% history record
            obj.histcnt = 0;
            obj.savecnt = 0;
            obj.uhist = [];                 %control history
            obj.xhist = [];                 %state history
        end
        
        %% set normalization matrix
        function md = setPara(md, name, value)
            if strcmp(name,'norm')
                md.xnorm = value;
            end
        end 
        
        %% reset the platform
        function md = reset(md, desire, xi)
            if nargin == 1              % reset state and control to initial value
                md.x = md.xi;
                md.u = md.ui;
            elseif nargin == 2          % reset desired value 
                md.x = md.xi;
                md.u = md.ui;
                md.desire = desire;
            elseif nargin == 3          % reset initial state
                md.xi = xi;
                md.x = md.xi;
                md.u = md.ui;
                md.desire = desire;
            end
        end
        
        %% simulate on the platform for unit period(dt) with action
        function [md,uc] = simulate(md, action)
            x0 = md.x;
            uc = action*md.uMag + md.unoise*(rand(1,numel(action)) - 0.5);      % add noise to the control signal
            xp = ode4(@invertPendulum, [0,md.dt], x0, uc);
            md.x = xp(end,:)';                                                  % get final state of simulation
            % save history
            md.xhist = [md.xhist, md.x];
            md.uhist = [md.uhist, uc];
            md.histcnt = md.histcnt + 1;
        end
        
        %% get state of the platform
        function [state, reward] = getState(md)
            state = md.x - md.desire;                               % [md.x(1); md.x(2)-md.desire];
            [reward]= md.reward(state);          % calculate reward and status of the platform
%             state = state./md.xnorm;                                % normalize the output, xnorm is one
            state = state';                                         
        end
        
        %% get reward and status of the platform     
        % status - 1 for succeed, -1 for failure, 0 for normal range
        function  [r]= reward(md,error)
            r = 0;
              r= error'*error;
        end
        
        %% show history of the state and control
        % clear history with clean equal to one
        function md = showHist(md, clean)
            if nargin == 1
                clean = 0;
            end
            figure(1);
            n = size(md.xhist,1) + size(md.uhist,1);
            for i = 1:n-1
            subplot(n,1,i);
            plot(md.xhist(i,:)');
            hold on;
            plot(ones(1,md.histcnt)*md.xnorm(i),'-r');
            plot(-1*ones(1,md.histcnt)*md.xnorm(i),'-r');
            title(md.xname(i));
            grid on;
            end
            subplot(n,1,n);
            plot(md.uhist);
            title('Control force');
            grid on;
%             figure(1);
%             plot(md.uhist);
%             figure(2);
%             plot(md.xhist');
%             legend(md.xname);
            if clean == 1
               %save([num2str(md.savecnt),'data.mat'],'md');
               md.uhist = [];
               md.xhist = [];
               md.histcnt = 0;
            end
            uiwait(gcf); % Wait for the figure to close before proceeding
        end
        
        %% get control list of the platform
        function [type, actionRange] = getControl(md)
            type = md.utype;                % control type is continous or discrete
            actionRange = md.ulist;         % control list or control range
        end
        
    end
end