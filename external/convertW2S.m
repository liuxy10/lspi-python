%% Convert W vector to symmetric matrix S
% This function takes a vector `W` and converts it into a symmetric matrix `S`.
% The input vector `W` is assumed to represent the upper triangular elements 
% of a symmetric matrix in column-major order. The function reconstructs the 
% full symmetric matrix by filling in the upper triangular part and then 
% symmetrizing it.
%
% Inputs:
%   - W: A column vector containing the upper triangular elements of a symmetric 
%        matrix, stored in column-major order.
%
% Outputs:
%   - S: A symmetric matrix reconstructed from the input vector `W`.
%
% Notes:
%   - The size of the symmetric matrix is determined by the variable `n`, which 
%     represents the number of rows/columns of the matrix.
%   - The function assumes that the length of `W` matches the number of elements 
%     in the upper triangular part of the symmetric matrix, i.e., length(W) = n*(n+1)/2.
%
% Example:
%   Given W = [1; 2; 3; 4; 5; 6] and n = 3, the function reconstructs:
%       Phat = [1 2 3;
%               0 4 5;
%               0 0 6];
%   The symmetric matrix S is then:
%       S = 1/2 * (Phat + Phat') = [1 2 3;
%                                   2 4 5;
%                                   3 5 6];
%% Convert W vector to S matrix
function S=convertW2S(W)
%             nQ = size(obj.Q,1);
%             nR = size(obj.R,1);
%             n = nQ+nR;
            n=4;
            idx = 1;
            for r = 1:n
                for c = r:n
                    Phat(r,c) = W(idx);
                    idx = idx + 1;
                end
            end
            S  = 1/2*(Phat+Phat');          
end

