%% Local basis function

% BASIS_QUADRATIC Generates a quadratic basis function for a given state and action.
%
%   phi = BASIS_QUADRATIC(state, action) computes a feature vector `phi` 
%   based on the quadratic combinations of the elements in the input 
%   `state` and `action`. The function concatenates the state and action 
%   into a single vector `z`, and then computes all quadratic terms 
%   (including cross terms and squared terms) from the elements of `z`.
%
%   Inputs:
%       state  - A vector representing the state.
%       action - A scalar or vector representing the action.
%
%   Outputs:
%       phi    - A column vector containing the quadratic basis features.
%
%   Notes:
%       - If no input arguments are provided, the function returns a 
%         default value of 10 for `phi`.
%       - The number of elements in the concatenated vector `z` is 
%         determined by the length of `state` and `action`.
%       - The quadratic terms are computed as z(r) * z(c) for all 
%         combinations of indices r and c, where r <= c.

function phi = basis_quadratic(state,action, n)

 if nargin < 1
    phi=6; % number of basis functions
    return
  end

z = cat(1,state,action);
idx = 1;
for r = 1:n
    for c = r:n
        if idx == 1
            phi = z(r)*z(c);
        else
            phi = cat(1,phi,z(r)*z(c)); % concatenate the quadratic terms
        end
        idx = idx + 1;
    end
end
end

