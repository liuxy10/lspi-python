%% Finds a good policy given a set of samples and a basis
function [policy, all_policies] = lspi(domain, maxiterations,epsilon, samples, basis, discount, initial_policy)
   %%% Initialize the random number generator to a random state
   
   rand('state', sum(100*clock));
  
  %%% Create a new policy
  initialize_policy = [domain + '_initialize_policy'];
  policy = feval(initialize_policy, 0.0, discount, basis);
  %%% Consider this to be the initial_policy if one is not provided
  if nargin<8
    initial_policy = policy;
  end

  
  %%% Initialize policy iteration 
  iteration = 0;
  distance = inf;
  all_policies{1} = initial_policy;
  
  
  %%% If no samples, return
  if ~isempty(samples)==0
    disp('Warning: Empty sample set');
    return
  end
  
  
  %%% Main LSPI loop  
  while  ((iteration<maxiterations)&&(distance > epsilon)) 
       
    %%% Update and print the number of iterations
    iteration = iteration + 1;
    disp('*********************************************************');
    disp( ['LSPI iteration : ', num2str(iteration)] );
    if (iteration==1)
      firsttime = 1;
    else
      firsttime = 0;
    end

      policy.weights = lsqfast(samples, all_policies{iteration}, ...
			       policy, firsttime);


    
    %%% Compute the distance between the current and the previous policy
    l1 = length(policy.weights);
    l2 = length(all_policies{iteration}.weights);
    if (l1 == l2)
      difference = policy.weights - all_policies{iteration}.weights;
      LMAXnorm = norm(difference,inf);
      L2norm = norm(difference);
    else
      LMAXnorm = abs(norm(policy.weights,inf) - ...
		     norm(all_policies{iteration}.weights,inf));
      L2norm = abs(norm(policy.weights) - ...
		   norm(all_policies{iteration}.weights));
    end
    distance = L2norm;
      
      
    
    %%% Print some information 
    disp( ['   Norms -> Lmax : ', num2str(LMAXnorm), ...
	   '   L2 : ',            num2str(L2norm)] );
    
    
    %%% Store the current policy
    all_policies{iteration+1} = policy;
    
    fprintf('Policy %d: weights: %s\n', iteration, mat2str(all_policies{iteration}.weights));%%% Depending on the domain, print additional info if needed
 %   feval([domain '_print_info'], all_policies);
    
  end
  
  
  %%% Display some info
  disp('*********************************************************');
  if (distance > epsilon) 
    disp(['LSPI finished in ' num2str(iteration) ...
	  ' iterations WITHOUT CONVERGENCE to a fixed point']);
  else
    disp(['LSPI converged in ' num2str(iteration) ' iterations']);
  end
  disp('********************************************************* ');

end

