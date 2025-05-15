%% Convert S matrix to W vector
function W=convertS2W(S)
               n=size(S,1);
%                nQ = size(obj.Q,1);
%                nR = size(obj.R,1);
%                nk = nQ+nR;
               nk=4;
               W=zeros(nk*(nk+1)/2,1);
               for i=1:n         
                 for j=i:n
                     m=(n+2-i+n)*(i-1)/2+j-i+1;
                     if j==i
                     W(m,1)=S(i,j);   
                     else
                      W(m,1)=2*S(i,j); 
                     end
                 end         
               end 
end

