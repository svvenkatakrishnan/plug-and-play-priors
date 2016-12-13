function out = plowColor(y, max_iter,sigma2)
% PLOW method for color images.
% The function assumes the following inputs
%
% y:  Noisy RGB image
% max_iter: Flag for pre-filtering [Optional input, default 1: No pre-filtering; 2: pre-filtering]
% sigma2: Variance of the noise in each color channel (1x3 vector for R, G, B) [Optional input]
%
% Output is as follows:
% 
% out: [h x w x 3 x max_iter] matrix, containing RGB images
%
% This file makes use of plowFastMex function which can be memory heavy

nC = feature('numCores');
setenv('OMP_NUM_THREADS', num2str(nC)); % utilize multiple cores
getenv OMP_NUM_THREADS

% Set variables
y2 = double(y);
[h w col] = size(y2);
sz = h*w;
ksz = 11; % Size of patches ksz x ksz
rad = (ksz-1)/2; % Radius for padding
wsz = 30; % Search windows size - 1
wrad = (wsz)/2;
d = ksz^2; % Dimensionality of patches
err = (5*2.55*ksz)^2;  % Threshold for unobserved noise-free patches

K=15; % Number of clusters
p=10; % Max number of patches to consider in similarity search

if(~exist('max_iter','var'))
    max_iter = 1;
end


% Create output variables
zy = zeros(h,w,max_iter);
zcb = zy;
zcr = zy;
z = y2;

if(~exist('sigma2','var')) % Estimate noise variance if no guide given
    sigma2 = zeros(1,3);
    
    for col=1:3
        y3 = z(:,:,col);
        y3 = conv2(y3, [2 -1; -1 0], 'same')./sqrt(6);
        sigma2(col) = (1.4826*median(abs(y3(:) - median(y3(:)))))^2;
    end
%        display(strcat('Estimated noise std :',num2str(sqrt(sigma2))));
    clear y3;
end


for iter = 1:max_iter
    
    for col = 1:3 % Process each color channel independently
                
        if( iter > 1)
            % noise estimate
            y3 = z(:,:,col);
            y3 = conv2(y3, [2 -1; -1 0], 'same')./sqrt(6);
            sig2_act = (1.4826*median(abs(y3(:) - median(y3(:)))))^2;
     %       display(strcat('Estimated noise std :',num2str(sqrt(sig2_act))));
            clear y3;
            
            
            sig2 = sig2_act;
            prepFlag = 0;           
            
        else
            
            sig2_act = sigma2(col);
            
           if(max_iter>1) % First iteration is pre-filtering
                sig2 = 0.75*sig2_act;
                prepFlag = 1;
           else
               sig2 = sig2_act;
               prepFlag = 0;
           end
        end
        
        
        if(col == 1) % Cluster only intensity image
            
            ztmp = RGB2YCC(z);
            %ztmp = z;
            
            %W = getSKRWeights(ztmp(:,:,1),ksz,1.4); 
            y3 = padarray(ztmp(:,:,1), [rad+2 rad+2], 'symmetric');
            W = getLARKMex(y3,ksz,1.4); clear y3;
            W = reshape(W, [d sz]);
            W = W./repmat(sum(W),[size(W,1) 1]);
            D = princomp(W'); D = D(:,1:10);
            W = W'*D;
        
            %idx = kmeans(W, K, 'rep', 1,'EmptyAction','drop'); 
            [idx Cen] = kmeansMex(W', K); 
            clear W D ztmp;
 %           display('Clustering done');
        end
        
 %       display(strcat('Processing channel : ',num2str(col))); 
        I = im2col(padarray(y2(:,:,col),[rad rad],'symmetric'),[ksz ksz],'sliding');
        Y = im2col(padarray(z(:,:,col),[rad rad],'symmetric'),[ksz ksz],'sliding');
        
        mul = 1.0;
     
        
        if(sig2>400)
            thresh = err + 3*sig2_act*d;
        else
            thresh = err + 2*sig2_act*d;
        end
        
        %Cn = sig2*eye(ksz^2);
        Ce = zeros(d,h*w); % Error covariance matrix for aggregation step
        Ztmp = Ce; % tmp matrix to hold processed vectors
        
        for k=1:K
   
   
           % display(strcat('Processing cluster : ',num2str(k))); 
            lst = find(idx==k);
            csz = numel(lst);
            if(csz==0) % Empty cluster
                continue;
            end
       
           
             my = mean(Y(:,lst),2); % Cluster mean
             Cy = cov(Y(:,lst)'); % Cluster covariance, noisy
      
            [V D] = eig(Cy);
            D = diag(D);
            D = D - sig2;      
            D(D<=0) = 0.0001; % Shrink eigen values less than zero
            
            %lst = (idx(mask) == k); lst = mask(lst);
            %Yk = Y(:,lst);
            
            [ztmp ce] = plowMexFast(Y, I, h, w, my, V', D, lst-1, sigma2(col), thresh, prepFlag);
            Ztmp(:,lst) = reshape(ztmp, [d csz]);
            Ce(:,lst) = reshape(ce, [d csz]);

% This part is Matlab equivalent of what we are doing in plowMexFast function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
%             for i=1:csz
%            
%                 yi = Y(:,lst(i));
%                 [r, c] = ind2sub([h w],lst(i));
%                 rmin = max(1, r - wrad);
%                 cmin = max(1,c - wrad);
%                 rmax = min(h, rmin + wsz);
%                 cmax = min(w, cmin + wsz);
%                 [C, R] = meshgrid(cmin:cmax, rmin:rmax);
%            
%                       
%                 lst2 = sub2ind([h w],R(:),C(:)); 
%                 clear R C;
%                 [idx_nn p2 nndist2] = getNNc(Y(:,lst2),yi,thresh,p); 
%                 
%                 idx_nn = lst2(idx_nn);
%                 
%                 if(iter==1 && sig2>=400)
%                     wts = single(p2)*exp(-nndist2'./h2);%./sig2; 
%                 else
%                     wts = exp(-nndist2'./h2);%./sig2;
%                 end
%                 wts = wts./sigma2(col);
%                 
%                 wSum = sum(wts);  
%                 D2 = D.*wSum;
%                 
%                 errMat = V*diag(D./(D2+1))*V';
%                 Ztmp(:,lst(i)) = my + errMat*sum((I(:,idx_nn) - my(:,ones(p2,1)))*wts,2);
%                 Ce(:,lst(i)) = diag(errMat);
%                 
%             end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        end
        
	% Aggregation step
        z(:,:,col) = vec2Img(Ztmp,Ce,[h w],1); 
        
	% Save output in appropriate color channel        
        switch col
            case 1
                zy(:,:,iter) = z(:,:,col);
                
            case 2
                zcb(:,:,iter) = z(:,:,col);
            case 3
                zcr(:,:,iter) = z(:,:,col);
        end
        
                   
    end
    
end


out = zeros(h,w,3,max_iter);

% Form output matrix
 for iter = 1:max_iter
     out(:,:,1,iter) = zy(:,:,iter);
     out(:,:,2,iter) = zcb(:,:,iter);
     out(:,:,3,iter) = zcr(:,:,iter);
 end


end     
                
           
            
            
