function [cost]=Cost_QGMRF(x,params)

[Rows,Col,Slice]=size(x);

temp=0;
for k=1:Slice
for i=1:Rows
for j=1:Col
    
if(j+1 <= Col)%changed from 3 to Col 
temp=temp+params.filter(2,3,2)*(((abs(x(i,j,k)-x(i,j+1,k)))^params.p)/(1 + ((abs(x(i,j,k)-x(i,j+1,k)))/params.c)^(params.p-params.q)));
end

if(i+1 <= Rows)
    if(j-1 >=1 )
      temp=temp+params.filter(3,1,2)*((abs(x(i,j,k)-x(i+1,j-1,k)))^params.p)/(1 + ((abs(x(i,j,k)-x(i+1,j-1,k)))/params.c)^(params.p-params.q));
      
    end
    
    temp=temp+params.filter(3,2,2)*((abs(x(i,j,k)-x(i+1,j,k)))^params.p)/(1 + ((abs(x(i,j,k)-x(i+1,j,k)))/params.c)^(params.p-params.q));
    
    
    if(j+1 <= Col)
        temp=temp+params.filter(3,3,2)*((abs(x(i,j,k)-x(i+1,j+1,k)))^params.p)/(1 + ((abs(x(i,j,k)-x(i+1,j+1,k)))/params.c)^(params.p-params.q));
    
    end
    
    if(k+1 <= Slice)
        if(j-1 >=1 )
            temp=temp+params.filter(3,1,3)*((abs(x(i,j,k)-x(i+1,j-1,k+1)))^params.p)/(1 + ((abs(x(i,j,k)-x(i+1,j-1,k+1)))/params.c)^(params.p-params.q));
    
        end
        
        temp=temp+params.filter(3,2,3)*((abs(x(i,j,k)-x(i+1,j,k+1)))^params.p)/(1 + ((abs(x(i,j,k)-x(i+1,j,k+1)))/params.c)^(params.p-params.q));
    
        
        if(j+1 <= Col)
            temp=temp+params.filter(3,3,3)*((abs(x(i,j,k)-x(i+1,j+1,k+1)))^params.p)/(1 + ((abs(x(i,j,k)-x(i+1,j+1,k+1)))/params.c)^(params.p-params.q));
    
        end
    end
end

if(k+1 <= Slice)
    
   if(i-1 >= 1)
       if(j-1 >= 1)
            temp=temp+params.filter(1,1,3)*((abs(x(i,j,k)-x(i-1,j-1,k+1)))^params.p)/(1 + ((abs(x(i,j,k)-x(i-1,j-1,k+1)))/params.c)^(params.p-params.q));
    
       end
            temp=temp+params.filter(1,2,3)*((abs(x(i,j,k)-x(i-1,j,k+1)))^params.p)/(1 + ((abs(x(i,j,k)-x(i-1,j,k+1)))/params.c)^(params.p-params.q));
    
            
        if(j+1 <= Col)
            temp=temp+params.filter(1,3,3)*((abs(x(i,j,k)-x(i-1,j+1,k+1)))^params.p)/(1 + ((abs(x(i,j,k)-x(i-1,j+1,k+1)))/params.c)^(params.p-params.q));
    
        end
   end
   
        if(j-1 >= 1)
            temp=temp+params.filter(2,1,3)*((abs(x(i,j,k)-x(i,j-1,k+1)))^params.p)/(1 + ((abs(x(i,j,k)-x(i,j-1,k+1)))/params.c)^(params.p-params.q));
    
       end
            temp=temp+params.filter(2,2,3)*((abs(x(i,j,k)-x(i,j,k+1)))^params.p)/(1 + ((abs(x(i,j,k)-x(i,j,k+1)))/params.c)^(params.p-params.q));
    
            
        if(j+1 <= Col)
            temp=temp+params.filter(2,3,3)*((abs(x(i,j,k)-x(i,j+1,k+1)))^params.p)/(1 + ((abs(x(i,j,k)-x(i,j+1,k+1)))/params.c)^(params.p-params.q));
    
        end
end

end
end
end
cost=temp*params.filter_scaling;

   
           
   

    
    