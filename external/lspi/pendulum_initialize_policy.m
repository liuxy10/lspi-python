%% initialize policy,Creates and initializes a new policy
function policy = pendulum_initialize_policy(explore, discount, basis)
 
  policy.explore = explore;
  
  policy.discount = discount;

  policy.actions = 1;
  
  policy.basis = basis;
  
  k = feval(basis);
  
  %%% Initial weights 
  policy.weights = convertSW(full(sprandsym(3,1,[1 0.3 0.01]))); % OK
  % policy.weights = [1,0,0,.3,0, 0.01]'; % initial weights
  %policy.weights = convertS2W(full(sprandsym(3,1,10*[1 2 1]))); 
 % policy.weights = 100*(2*rand(6,1)-1);
%  policy.weights=[0.00533;0.0113;0.01176;0.0241;0.02499;0.02595];
%  policy.weights = [0.0064 -0.0054 -0.0027 0.0045 0.0023 0.0011]';
  %policy.weights = ones(k,1);  % Ones
  %policy.weights = rand(k,1);  % Random

end