%% lsqfast method returns the learned weights w and the matrices A and b
function [w, A, b] = lsqfast(samples, policy, new_policy, firsttime)
  persistent Phihat
  persistent Rhat
  %%% Initialize variables
  howmany = length(samples);
  k = feval(new_policy.basis); % 
  A = zeros(k, k);
  b = zeros(k, 1);
  PiPhihat = zeros(howmany,k);
  mytime = cputime;  
  
  %%% Precompute Phihat and Rhat for all subsequent iterations
  if firsttime == 1
    
    Phihat = zeros(howmany,k);
    Rhat = zeros(howmany,1);
    
    for i=1:howmany
      phi = feval(new_policy.basis, samples(i).state, samples(i).action, 3);
      Phihat(i,:) = phi';
      Rhat(i) = samples(i).reward;
    end
    
  end 
  
  %%% Loop through the samples 
  for i=1:howmany
    
    %%% Make sure the nextstate is not an absorbing state
   if ~samples(i).absorb
      
      %%% Compute the policy and the corresponding basis at the next state 
      %%%  [action, actionphi] = policy_function(policy, state)
      nextaction = policy_function(policy, samples(i).nextstate);
      nextphi = feval(new_policy.basis, samples(i).nextstate, nextaction, 3);
      PiPhihat(i,:) = nextphi';      
  end   
  end
  
  %%% Compute the matrices A and b
  A = (Phihat)' * (Phihat - new_policy.discount * PiPhihat);
  b = (Phihat)'* Rhat;
  
  % phi_time = cputime - mytime;
  % disp(['CPU time to form A and b : ' num2str(phi_time)]);
  % mytime = cputime;
   
  %%% Solve the system to find w
  rankA = rank(A);
  
  % rank_time = cputime - mytime;
  % disp(['CPU time to find the rank of A : ' num2str(phi_time)]);
  % mytime = cputime;
  
  disp(['Rank of matrix A : ' num2str(rankA)]);
%   if rankA==k
%       disp('A is a full rank matrix!!!');
%       w = A\b;
%   else
%       disp(['WARNING: A is lower rank!!! Should be ' num2str(k)]);
%       w = pinv(A)*b;
%   end
  
  
  j=1;
  error=1;
  stepsize=0.5;
  phiw=zeros(k,1);
  Cphi=A/howmany;
  dphi=b/howmany;
  while j<=1000 && error>1e-4 %(j<=ns for offline training)
      oPhiw=phiw;
      residuePhi=phiw-stepsize*(Cphi*phiw-dphi);
      phiw=proDysktra(residuePhi,100,1e-4);
      j=j+1;
      error=norm(oPhiw-phiw);
      stepsize=1/(j+1);
      %                    stepsize=1e-3;
  end
  w=phiw;


solve_time = cputime - mytime;
disp(['CPU time to solve Aw=b : ' num2str(solve_time)]);
end

