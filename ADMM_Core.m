function [map_image,params]=ADMM_Core(map_image,sino,geom,Amatrix,params,kparams,mode)
%Performs ADMM based minimization of min_{x,v} (1/2)||y-Ax||^2 + s(v)
% Inputs
% map_image is an inital estimate for x
% sino is a structure which contains all the sinogram information
% geom contains the geometry information like number of pixels etc.
% Amatrix = A stored as a sparse matrix
% params contains parameters associated with the optimization
% kparams contains prior model parameters
% mode represents what prior is used in s(v)
%    0 - kSVD
%    1 - BM3D
%    2 - TV
%    3 - PLOW
%    4 - qGGMRF
%    5 - MAP segmentation

iter=1;
stop_crit=1;
[m n]=size(map_image);

%% DEBUG STATEMENTS
if(params.verbose)
    %Debug code - cost computation for qGGMRF - ICD
    if(mode == 4 && kparams.verbose == 1)
        cost(iter)=0;
        [sinonew]=forward_project_v2(map_image,sino,Amatrix);
        [m_s n_s]=size(sino.counts);
        D = zeros(m_s,n_s);
        for i=1:m_s
            for j=1:n_s
                if(sino.counts(i,j)~=0)
                    D(i,j)=1/sino.counts(i,j);
                else
                    D(i,j) =0;
                end
            end
        end
        cost(iter)=(1/2)*sum(D(:).*(sino.counts(:)-sinonew(:)).^2);
        kparams.filter=params.filter;
        cost(iter)=cost(iter)+(params.beta*kparams.p*kparams.sigmax^kparams.p)*qGGMRFprior2DCost(map_image,kparams);
    end
    RMSE(iter) = sqrt(sum((sum((map_image - params.original).^2)))/numel(params.original));
end


%% END OF DEBUG STATEMENTS

while (iter < params.max_iter && stop_crit > params.threshold)
    
    if(params.verbose)
        display(iter);
    end
    
    if(iter > 1) %After the first outer iteration just do 1 iteration
        params.num_iter=1;
        
        if(isfield(kparams,'initdict') && ischar(kparams.initdict))
            kparams.initdict=dict; %for kSVD use the previous dictionary to start
        end
    end
    
    map_image = DataTermOpt(map_image,sino,geom,params,Amatrix);
    kparams.x = map_image + params.u;
    
    switch(mode)
        
        %%ksvd
        case 0
            kparams.x = (kparams.x./kparams.maxval).*255;%Normalize to 255
            [imout,dict]=ksvddenoise(kparams,0);
            imout = (imout./255).*kparams.maxval;%Normalize this back to the regular scale
        case 1
            %%bm3D ; maxval is the max of noisy images (approximately)
            [NA,imout] = BM3D(1,kparams.x/kparams.maxval,kparams.sigma);
            imout=imout.*kparams.maxval;%FOR BM3D this is needed       
        case 2
            %TV denoising
            [imout,err,tv,lambda] = perform_tv_denoising(kparams.x,kparams);
            %imout=imout.*255;
        case 3
            kparams.x = (kparams.x./kparams.maxval).*255;%Normalize to 255
            Output = plowMatlab(zeros(m,n), kparams.x, kparams.sigma);
            imout=Output(:,:,end);
            imout = (imout./255).*kparams.maxval;%Normalize this back to the regular scale
        case 4
            imout = qGGMRFdenoise(kparams.x,kparams);
        case 5
            H=hamming(3)*hamming(3)';
            H=H./sum(H(:));
            Initial = imfilter(kparams.x,H);
            [Temp,~] = otsu(Initial,kparams.num_class);%Get a initial label set from Otsu
            [imout,~,~]=MAP_segmentation(kparams.x,kparams,Temp);
    end
    
    prev_v=params.v;
    params.v = imout;
    params.u = params.u+(map_image-params.v);
    
    eps_primal = sum(abs(params.v(:)-map_image(:)))./sum(abs(map_image(:)));    
    eps_dual = sum(abs(params.v(:)-prev_v(:)))./sum((abs(prev_v(:))));
    
    stop_crit = (eps_primal+eps_dual)/2;
    
    iter=iter+1;
    
    if(params.verbose==1)
        display(stop_crit);
        
        if(mode == 4 && kparams.verbose == 1)
            %Debug code - cost computation for qGGMRF
            cost(iter)=0;
            [sinonew]=forward_project_v2(map_image,sino,Amatrix);
            cost(iter)=(1/2)*sum(D(:).*(sino.counts(:)-sinonew(:)).^2);
            cost(iter)=cost(iter)+(params.beta*kparams.p*kparams.sigmax^kparams.p)*qGGMRFprior2DCost(map_image,kparams);
            
            if(cost(iter)-cost(iter-1)>0)
                display('Cost just increased!');
            end
        end
        
        %Calculate RMSE
        RMSE(iter) = sqrt(sum((sum((map_image - params.original).^2)))/numel(params.original));
    end
    
end

if(params.verbose==1)
    
    if(mode == 0) %kSVD display final dictionary
        dictimg = showdict(dict,[1 1]*kparams.blocksize,round(sqrt(kparams.dictsize)),round(sqrt(kparams.dictsize)),'lines','highcontrast');
        figure; imshow(imresize(dictimg,2,'nearest'));
        title('Trained dictionary');
        display(iter);
    end
    if(mode == 4 && kparams.verbose == 1) %qGGMRF debug
        figure;
        plot(cost);
        cost_admm=cost;
        RMSE_ADMM=RMSE;
        save('ADMMCostv2.mat','cost_admm','RMSE');
    end
    params.RMSE=RMSE;
end

