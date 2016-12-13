function [new_image]=DataTermOpt(map_image,sino,geom,params,Amatrix)
%Optimizes the function : data mismatch term + augmented vector
% min_x ((1/2)||y-Ax||_D^{2} + lambda ||x-(v-rho)||^2 )
%Inputs: map_image : Initial value of x 
%        sino : sinogram structure containing data
%        geom : geometry structure contianing geom info of object 
%        params : prior model parameters + algorithm parameters structure
%        Amatrix : A stored as a userdefined sparse vector 

[m n]=size(map_image);

Ax=forward_project_v2(map_image,sino,Amatrix); %returns a n_theta X n_t matrix

%if(isfield(params,'xray') && params.xray == 1)
%    sino.counts=-log(sino.counts./params.dose);
%end

e=(sino.counts-Ax)'; %n_t X n_theta
e=reshape(e,1,sino.n_t*sino.n_theta);


if(params.verbose)
cost = zeros(1,params.num_iter);
end


if(~isfield(params,'xray') || params.xray == 0) %if this is not transmission data use the corret noise model
    d=reshape(sino.counts',1,sino.n_t*sino.n_theta);%Diagonal covariance matrix entries
    for i=1:length(d)
        if(d(i)~=0)
            d(i)=1/d(i);
        else
            d(i)=0;
        end
    end
else
    if(params.xray==1)
        d=reshape(sino.weight',1,sino.n_t*sino.n_theta);%Diagonal covariance matrix entries
    end
end

%TempK = params.v-params.u; %A temporary variable used to store v-u

for k=1:params.num_iter
    
    %computing cost after each iteration
    if(params.verbose)
        cost(k) = Cost_DataTermAL(e,d,map_image,params);
    end
    prev_img = map_image;
    
    [map_image,e]=HomogenousUpdate(map_image,Amatrix,e,d,params);
    
%     if(mod(k,2) ~= 0) %alternate between homogenous and non-homogenous updates
%         [map_image,e]=HomogenousUpdate(map_image,Amatrix,e,d,params);
%     else
%         K=20;
%         [map_image,e]=NonHomogenousUpdate(map_image,Amatrix,e,d,params,K,UpdateMap);
%     end
    
    UpdateMap=abs(map_image-prev_img);
       
end

new_image = map_image;





