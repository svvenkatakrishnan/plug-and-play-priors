function [SegImage,ClassLabel,params]=MAP_segmentation(InpImg,params,InitClassLabel)
%2-D MAP segmentation and mapping
%Takes a noisy image and produces a discrete valued output
%min{b,x(b)} (1/2\sigma^2)||y-x(b)||_{2}^2 + \beta * s(b)
%Inputs: InpImg  : Noisy input image y in above equation
%        params : A structure which specifies parameters associated with
%        the prior model as well as ICD algorithm
%               params.filter : a 3 X 3 weighting filter for the prior
%               params.num_iter : Max number of iterations
%               params.beta : regularization parameter
%               params.sigma_sq : Noise variance 
%
%       InitClassLabel : An initial label of class labels from 1...K where
%       K is the total number of classes in the image i.e.
%       K = max(InitClassLabel(:))
%Outputs: SegImage : A image with a discrete set of levels
%         ClassLabel : A image of class labels from 1 to K
[m n]=size(InpImg);
ClassLabel = InitClassLabel;
K = max(ClassLabel(:)); %Number of classes
mu=zeros(1,K);%Class means
sigma_sq = zeros(1,K);%Class variance


%Initialize the class means and variance based on the initial labels and
%input image
[SegImage mu sigma_sq]=ClassLabel2GrayScale(InpImg,ClassLabel,K);
sigma_sq = params.sigma_sq.*ones(1,K);

InvVarImg = zeros(m,n);
for LabelIdx=1:K
    InvVarImg(ClassLabel==LabelIdx)=1/sigma_sq(LabelIdx);
end

if(params.debug)
    cost(1) = (1/2).*sum(InvVarImg(:).*(InpImg(:)-SegImage(:)).^2) + 0.5*sum(log(2*pi*(1./InvVarImg(:)))) + SegmentationPrior2DCost(ClassLabel,params);
end

for Iter=1:params.max_iter   %Outer Loops
    
    
    change=0;%debugging variable to track how many labels change
    
    if(isfield(params,'rand_ord') && params.rand_ord == 1)
        %Initialization for randomized ICM
        RandArray=1:m*n;
        CurLenRandArray = m*n;
    end
    
    for i=1:m
        for j=1:n
            
            if(isfield(params,'rand_ord') && params.rand_ord==1)
                
                r=randi(CurLenRandArray,1);
                temp_idx = RandArray(r)-1;
                
                %Swap two elements
                temp = RandArray(CurLenRandArray);
                RandArray(CurLenRandArray) = RandArray(r);
                RandArray(r) = temp;
                
                %Decrement size of rand array
                CurLenRandArray=CurLenRandArray-1;
                
                r_new = mod(temp_idx,n)+1;
                c_new = floor(temp_idx/n)+1;
            else
                
                %Current label of pixel at (i,j)
                r_new=i;
                c_new=j;
            end
            
            CurrLabel = ClassLabel(r_new,c_new);
            
            %Find optimal class label for a given pixel
            TempArray = zeros(1,K);
            for k=1:K
                TempArray(k) = 0.5*(((InpImg(r_new,c_new)-mu(k))^2)/sigma_sq(k) + log(sigma_sq(k)));%forward model cost
                TempSum=0;
                for p=-1:1
                    for q=-1:1
                        if(r_new+p >= 1 && r_new+p <= m && c_new+q >=1 && c_new+q <= n)
                            if(ClassLabel(r_new+p,c_new+q) ~= k)
                                TempSum = TempSum + params.filter(p+2,q+2);
                            end
                        end
                    end
                end
                TempSum = TempSum*params.beta;
                TempArray(k) = TempArray(k) + TempSum;
            end
            [a NewLabel] = min(TempArray);
            if(CurrLabel ~= NewLabel)%Reassign the class means and variances in case of a change
                change=change+1;
                [mu sigma_sq] = UpdateSegClassParams(mu,sigma_sq,[CurrLabel NewLabel],ClassLabel,InpImg(r_new,c_new));
                
                sigma_sq = params.sigma_sq.*ones(1,K);
                
                ClassLabel(r_new,c_new)=NewLabel;
            end
        end
    end
    %display(change);
    if(params.debug)
        for LabelIdx=1:K
            SegImage(ClassLabel==LabelIdx)=mu(LabelIdx);
        end
        cost(Iter+1) = (1/2).*sum(InvVarImg(:).*(InpImg(:)-SegImage(:)).^2) + 0.5*sum(log(2*pi*(1./InvVarImg(:)))) + SegmentationPrior2DCost(ClassLabel,params);
    end
    
end
if(params.debug)
    params.cost=cost;
    figure;
    plot(cost);
end

[SegImage,mu,sigma_sq]=ClassLabel2GrayScale(InpImg,ClassLabel,K);