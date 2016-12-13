function [map_image,e]=HomogenousUpdate(map_image,Amatrix,e,d,params)
%Function that performs random order ICD with a quadratic penalty 
%on the input (for Augmented lagrangian)
%Inputs map_image : initial value of image to be reconstructed
%       Amatrix   : A matrix 
%       e : current error vector 
%       d : diagonal weights vector
%       params: hold the parameter for Augmented Lagrangian variables
% Outputs : map_image : Final reconstruction
%           e : Updated error vector 

TempK = params.v-params.u;

[m n]=size(map_image);
%Initialization for randomized ICD

RandArray=1:m*n;

CurLenRandArray = m*n;

for row=1:m
    for col=1:n
        
        r=randi(CurLenRandArray,1);
        temp_idx = RandArray(r)-1;
        
        %Swap two elements
        %temp = RandArray(CurLenRandArray);
        %RandArray(CurLenRandArray) = RandArray(r);
        %RandArray(r) = temp;
        RandArray(r)=RandArray(CurLenRandArray);%TODO - modifed to simply a move instead of a swap
        
        %Decrement size of rand array
        CurLenRandArray = CurLenRandArray-1;
        
        r_new = mod(temp_idx,m)+1; %TODO: Changed to "m" --8/19/2013
        c_new = floor(temp_idx/m)+1;
        
%         if(r_new < 1 || r_new > n)
%             display('Bug in random routine');
%             display(r_new);
%         end
%         if(c_new < 1 || c_new > m)
%             display('Bug in random routine');
%             display(c_new);
%         end
        
        neighborhood=zeros(3,3);
        p=1;
        for y=r_new-1:r_new+1
            q=1;
            for z=c_new-1:c_new+1
                if(y >= 1 && z >=1 && y<=m && z<=n)
                    neighborhood(p,q)=map_image(y,z);
                end
                q=q+1;
            end
            p=p+1;
        end
        neighborhood(2,2)=0;
        
        v=map_image(r_new,c_new);
        
        
        L=length(Amatrix(r_new,c_new).value);
        theta1=0;
        theta2=0;
        
%        theta2 = Amatrix(r_new,c_new).value*diag(d(Amatrix(r_new,c_new).index))*Amatrix(r_new,c_new).value';
%        theta1 = e(Amatrix(r_new,c_new).index)*diag(d(Amatrix(r_new,c_new).index))*Amatrix(r_new,c_new).value';

        theta2 = Amatrix(r_new,c_new).value*((d(Amatrix(r_new,c_new).index)').*Amatrix(r_new,c_new).value');
        theta1 = e(Amatrix(r_new,c_new).index)*(d(Amatrix(r_new,c_new).index)'.*Amatrix(r_new,c_new).value');
        
%         for i=1:L
%             theta2=theta2 + (Amatrix(r_new,c_new).value(i))^2 *d(Amatrix(r_new,c_new).index(i));
%             theta1=theta1 + e(Amatrix(r_new,c_new).index(i))*Amatrix(r_new,c_new).value(i)*d(Amatrix(r_new,c_new).index(i));
%         end
               
        temp = (theta1 - (v-TempK(r_new,c_new))*params.lambda)/(theta2 + params.lambda);
        update = v + temp;
        if(update > 0)
            map_image(r_new,c_new) = update;
        else
            map_image(r_new,c_new) = 0;
        end
        
        
       % for i=1:L
       %     e(Amatrix(r_new,c_new).index(i))=e(Amatrix(r_new,c_new).index(i))-Amatrix(r_new,c_new).value(i)*(map_image(r_new,c_new)-v);
       % end
       
       e(Amatrix(r_new,c_new).index)=e(Amatrix(r_new,c_new).index)-Amatrix(r_new,c_new).value*(map_image(r_new,c_new)-v);
       
    end
end