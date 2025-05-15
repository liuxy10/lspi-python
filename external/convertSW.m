        %% Convert S matrix to W vector
        function W=convertSW(S)
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
