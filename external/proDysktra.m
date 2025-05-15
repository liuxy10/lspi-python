%% alternating projection
function x=proDysktra(x0,ballR,errTol)           
            error=1;
            j=1;
            I=zeros(length(x0),2);
            oldI=zeros(length(x0),2);
            x=x0;
            while j<500 && error>errTol 
                
                oldX=x;
                if norm(x-I(:,1))>ballR
                    x=ballR*(x-I(:,1))/norm(x-I(:,1));                    
                else
                    x=x-I(:,1);
                end                
                oldI(:,1)=I(:,1);
                I(:,1)=x-(oldX-I(:,1));
                
                oldX=x;
                s=convertWS(x-I(:,2));
                [V,D] = eig(s); % D is diagonal matrix, V is orthogonal
                D(D<0)=0;                              
                s=V*D*V';
                x=convertSW(s); % x is the new point
                oldI(:,2)=I(:,2);
                I(:,2)=x-(oldX-I(:,2));  
                                
                j=j+1;
                error=norm(oldI-I)^2;                
            end            
end

