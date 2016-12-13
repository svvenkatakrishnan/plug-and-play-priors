function [ImOut] = qGGMRFdenoise(ImInput,params)
%Function to denoise a image using a q-Generalized MRF prior
% (1/2*\sigma^2)||y-x||_{2}^2 + (1/p*sigmax^p)\sum_{i,j} filter_{ij}*rho(delta_{i,j})
% Inputs: ImInput - the noisy image
%        params.sigma  - noise variance estimate/ regularization
%        params.p, params.q, params.c,params.sigmax - qGGMRF parameters
%        params.niter - max number of iterations
%        params.verbose - if this is 1 display debug messages else suppress
%                         outputs
% Output: ImOut - the denoised image
[m n]=size(ImInput);
x=ImInput;
for k =1:params.niter
    
    if(params.verbose == 1)
        display(k);
    end
    
    %computing cost after each iteration
    %Initialization for randomized ICD
    RandArray=1:m*n;
    CurLenRandArray = m*n;
    
    for i=1:m
        for j=1:n
            
            %Generating a randomized index
            r=randi(CurLenRandArray,1);
            temp_idx = RandArray(r)-1;
            
            %Swap two elements
            temp = RandArray(CurLenRandArray);
            RandArray(CurLenRandArray) = RandArray(r);
            RandArray(r) = temp;
            
            %Decrement size of rand array
            CurLenRandArray=CurLenRandArray-1;            
            
            %Turn linear index to 2-D array index
            r_new = mod(temp_idx,n)+1;
            c_new = floor(temp_idx/n)+1;
            
            location=[r_new c_new];
            theta1 = x(r_new,c_new)-ImInput(r_new,c_new);
            theta2 = 1;
            theta1=theta1/(params.sigma^2);
            theta2=theta2/(params.sigma^2);
            u = Solve(x,theta1,theta2,params.sigmax,params.p,params.q,params.c,location);
            x(r_new,c_new)=u;
        end
    end
end
ImOut=x;

if(params.verbose == 1)
    figure;imagesc(ImInput,[0 255]);axis image;
    format_image_for_publication(gcf);
    colormap(gray);colorbar('Eastoutside');
    title('Noisy image');
    
    figure;imagesc(ImOut,[0 255]);axis image;
    format_image_for_publication(gcf);
    colormap(gray);colorbar('Eastoutside');
    title('Denoised image');
end