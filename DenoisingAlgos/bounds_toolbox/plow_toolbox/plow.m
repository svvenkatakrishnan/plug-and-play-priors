function z = plow(img,y,sigma,skip)

%%% Function to denoise patches by estimating the parameters of the PLOW
% filter in each cluster, and the photometric weights for each patch 
%
% Written by Priyam Chatterjee
% Last modified: May 23, 2011
%
% P. Chatterjee & P. Milanfar, "Patch-based Near-Optimal Image Denoising",
% IEEE TIP, vol. 21, num. 4, Apr. 2012.
%
%
% Input:
% img:      Noise-free image used to show MSE (optional)
% y:        Noisy image (required) : size hxw
% sigma:    Noise standard deviation (optional but suggested, if known)
% skip:     Pixel shifts for overlapping patches (default: 1, recommended <= 3)
%           1: implies densely overlapping patches, larger values make process faster

% z:        output image(s). If pre-processing required, z is h x w x 2
% idx:      cluster index of each patch (for debugging)
% Ztmp:     output of PLOW before aggregation (for debugging)
%
% Usage:
% [z idx Ztmp] = plow(img, y, sigma); -- Use default skip with ground-truth
% z = plowFast([],y); -- Default skip, no ground-truth or noise estimate
% z = plowFast([],y,sigma); -- no ground-truth, but prior info about noise
%
%%%

nC = feature('numCores');
setenv('OMP_NUM_THREADS', num2str(nC)); % utilize multiple cores
%getenv OMP_NUM_THREADS



if(~exist('skip','var'))
    skip = 1;
end

y2 = y;
[h w] = size(y);
ksz = 11;
rad = (ksz-1)/2;
wsz = 30; % Search windows size - 1
wrad = (wsz)/2;

d = ksz^2;
sz = h*w;

sig2 = sigma^2;


gamma = (5*2.55*ksz)^2;

K=15;

y = padarray(y, [rad rad], 'symmetric');

if(sigma>=10)
    max_iter = 2; % Set prefiltering on
else
    max_iter = 1;
end
z = zeros(h,w,max_iter);
prepFlag = 0;


for iter =1:max_iter
    
    if(iter>1 )

        y2 = conv2(z(:,:,iter-1),[2 -1; -1 0],'same')./sqrt(6);
        sig2 = (1.4826*median(abs(y2(:) - median(y2(:)))))^2; %MAD estimate
        y2 = z(:,:,iter-1);
    
        display(strcat('Using estimated std dev : ',num2str(sqrt(sig2))));
        sig2_act = sig2;
        prepFlag = 0;
        
    else
         
         sig2_act = sig2;
         sig2 = 0.75*sig2; % Leave some noise behind in initial iteration
         if(sig2 >= 400)
             prepFlag = 1;
         end
    end
    
    
    %%% Obtain the patch features for clustering    
    
    %W = getLARK(y2,ksz,1.4);  % Function in Matlab, equiv to next 3
                                     % lines for faster Mex function
    y3 = padarray(y2, [rad+2 rad+2], 'symmetric');
    W = getLARKMex(y3, ksz, 1.4); clear y3;
    W = reshape(W, [d sz]);
    
    W = W./repmat(sum(W),[size(W,1) 1]);
    D = princomp(W'); D = D(:,1:10);
    W = W'*D;
    
    % Perform clustering
    [idx Cen] = kmeansMex(W',K); clear Cen; % Simpler but faster Mex code
    %idx = kmeans(W,K,'rep',1,'EmptyAction','drop'); % Equiv Matlab function
    clear W W2 D;
    
    
    % If noise is a little strong, use a bigger theshold
    if(sig2>200)
        thresh = gamma + 3*sig2_act*d;
    else
        thresh = gamma + 2*sig2_act*d;
    end
    
    
    Ce = zeros(d,h*w);
    Ztmp = Ce;
    
    y2 = padarray(y2, [rad rad], 'symmetric');
    Y = im2col(y2, [ksz ksz], 'sliding');
   
   for k=1:K
   
    %   display(strcat('Processing cluster : ',num2str(k))); 
       lst = find(idx==k);
       csz = numel(lst); 
       
       if(csz==0) % Empty cluster
           continue;
       end
      
             
      my = mean(Y(:,lst),2);
      Cy = cov(Y(:,lst)');
      
      [V D] = eig(Cy);
      D = diag(D);
      
      
        D = D - sig2; 
      
      D(D<=0) = 0.0001;
      
      [Ridx, Cidx] = ind2sub([h w], lst); 
       
%        tic;
%         [ztmp ce] = plowM(y2, y, my, V, D, Ridx, Cidx, sigma^2, ksz, thresh, prepFlag);
%         toc;
       
      % Since C indexes from 0 -> Ridx = Ridx-1;
      % Since C reads column-wise -> V = V';
      %my = zeros(d,1,'double');
   %   display(strcat('Size of cluster : ',num2str(csz)));
   %   tic;
      [ztmp ce] = plowMex(y2, y, my, V', D, Ridx-1, Cidx-1, sigma^2, ksz, thresh, prepFlag);
   %    toc;
       
       Ztmp(:,lst) = reshape(ztmp, [d csz]);
       Ce(:,lst) = reshape(ce, [d csz]);
   end
   clear Y ztmp ce;
   z(:,:,iter) = vec2Img(Ztmp,Ce,[h w],skip);
   clear Ztmp Ce;
   
   
   ztmp = z(:,:,iter);
   if(exist('img','var'))
       display(strcat('MSE : ', num2str(mean2((img - ztmp).^2))));
       ssim = ssim_index(img, ztmp);
       display(strcat('SSIM : ', num2str(ssim)));
   end
   
end

