        %% Convert W vector to S matrix
        function S=convertWS(W)            
         nSA = round((-1 + sqrt(1 + 8 * length(W)))/2); 
         if size(W,2)>1
               fprintf('dimension error: W can only be a column vector \n');
            elseif size(W,1)~=(nchoosek(nSA,2)+nSA)
               fprintf('dimension error: W has a wrong row number \n'); 
            end

            bu=triu(ones(nSA),0); 
            bu=bu';
            bu(bu==1)=W;
            bu=bu';

            bl=tril(ones(nSA),0); 
            bl(bl==1)=W;

            S=bu+bl;
            S = S - 0.5*diag(diag(S));     
        end

