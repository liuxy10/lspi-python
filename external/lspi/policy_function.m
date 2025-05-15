%% Computes the "policy" at the given "state".
% Returns the "action" that the policy picks at that "state".
% Inputs:
%   - policy: A structure containing policy parameters (e.g., weights, exploration rate).
%   - state: The current state for which the policy is evaluated.
% Outputs:
%   - action: The action selected by the policy.
%   - actionphi: The basis function evaluated at the given state and action.

function [action, actionphi] = policy_function(policy, state)
  %%% Exploration or not?
% Exploitation: Use the policy weights to compute the optimal action.
  nS =2; % Number of state variables.
  nSA = nS + policy.actions; % Number of state and action variables.

  % Decide whether to explore or exploit based on the exploration rate.
  if (rand < policy.explore)
    sample_action_LE = sort(2 * rand(1, policy.actions) - 1); % Generate two random values in [-1, 1].
    action = sample_action_LE * 0.12;
    actionphi = feval(policy.basis, state, action, nSA);
  else
    
    
    % Convert policy weights into a structured form.
    HW = convertWS(policy.weights);
    Hux = HW(nS+1:nSA, 1:nS); % Cross-term between actions and states.
    Huu = HW(nS+1:nSA, nS+1:nSA); % Quadratic term for actions.
    if isempty(Huu) || all(Huu(:) == 0)
        Huu = eye(size(Huu, 1)); % Initialize Huu as an identity matrix if empty or all zeros.
    end
    Hxx = HW(1:nS, 1:nS); % Quadratic term for states.
    Hxu = HW(1:nS, nS+1:nSA); % Cross-term between states and actions.
    
    % Compute the linear term for the optimization problem.
    Hf = state' * Hxu;
    
    % Set up optimization constraints.
    options = optimset('display', 'off'); % Suppress optimization output.
    % A = double([-1 0; 1 0; 0 -1; 0 1]); % Action bounds.
    % Ax = double([(-0.01 + y1); (0.5 - y1); -0.5 + y2; 0.98 - y2]); % State-dependent bounds.
    
    % Solve the quadratic programming problem to find the optimal action. 
    % The optimization goal is to minimize: (1/2) * action' * Huu * action + Hf * action
    % action = quadprog(double(Huu), double(Hf'), double(A), double(Ax), [], [], [-1; -1], [1; 1], [], options);
    % Adjust the bounds to match the length of the decision variable 'action'.
    lb = [-1; -1]; % Lower bounds for the action.
    ub = [1; 1];   % Upper bounds for the action.
    lb = lb(1:size(Huu, 1)); % Ensure bounds match the size of Huu.
    ub = ub(1:size(Huu, 1)); % Ensure bounds match the size of Huu.
    
    % Solve the quadratic programming problem with adjusted bounds.
    action = quadprog(double(Huu), double(Hf'), [], [], [], [], lb, ub, [], optimset(options, 'Display', 'off'));

    % Add exploration noise to the action based on the RL iteration count.
    % action(1) = action(1) + max(0, (1 - RL_number / 3)) * (2 * rand - 1) * 0.15;
    % action(2) = action(2) + max(0, (1 - RL_number / 3)) * (2 * rand - 1) * 0.15;
  % ends

  % % Ensure the action satisfies additional constraints.
  % if (action(1) + y1 < 0.01)
  %   action(1) = 2 * abs(y1 - 0.01);
  % end

  % if ((y2 + action(2) - y1 - action(1)) < 0)
  %   action(2) = abs(y2 - y1 - action(1)) * 2;
  % end

  % if (action(2) + y2 > 0.98)
  %   action(2) = -2 * abs(0.98 - y2);
  % end

  % Compute the basis function for the given state and action.
  actionphi = feval(policy.basis, state, action, nSA);
end
