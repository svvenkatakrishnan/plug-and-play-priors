% Plug and Play priors script file
% To solve (1/2)||y-Ax||_{D}^2+ beta*prior(x)
% sino - Tomographic projection data ntheta X nr
% geom - object dimensions
% lambda - lagrange multiplier
% u - augmented lagrangian vector nx X ny (Object size)
% v - temporary image used in AL method nx X ny
clc;
clear;
close all;

load('data/AL_Data.mat'); %assumes this file has the tomography data stored appropriately 


%Image attributes
ScaleMin = 0;
ScaleMax = 255;
[m n]=size(Object);
ObjectSize = max(m,n);


%Choose the denoising algorithm 
% 0 - kSVD
% 1-BM3D
% 2-TV
% 3-PLOW
% 4-qGGMRF 
% 5- MAP segmentation
mode = 1;
 
params.max_iter = 100; %max outer iterations for ADMM
params.threshold = 1e-3;% Stopping criteria for algorithm
params.lambda = 1/75 ;%Lagrange multiplier
params.beta = 1.5; %The regularization parameter 
params.verbose=1; %outputs for debugging 0 -off/1-on
Display = 1; %Displays the output images

%% Other paramters for inner optimization - based on coordinate descent
params.num_iter=10;%Number of ICD inner loops per ADMM iter
params.filter=[1/12 1/6 1/12;1/6 0 1/6;1/12 1/6 1/12]; %TODO : Remove 
params.u = zeros(m,n); %Augmented Lagrange vector
params.v   = zeros(m,n); %Auxilarry variable
params.original=Object;%Phantom used for RMSE computes

%% Denoising parameter for ALL algorithms
kparams.sigma=sqrt(params.beta/params.lambda);

%% Denoising parameters - KSVD/BM3D 
if(mode == 0 || mode == 1)
    %kparams.x=(map_image./max(map_image(:))).*255;
    kparams.blocksize = 4;
    kparams.dictsize = 512;%256
    kparams.maxval= ScaleMax;
    kparams.trainnum = 3600;
    kparams.iternum = 10;
    kparams.memusage='high';
end
%% These are only for TV denoising
if(mode == 2)
    kparams.verb = 0;
    kparams.display = 0;
    kparams.niter = 100;    % number of iterations
    kparams.c_TV = .0998;
    kparams.lambda = 2*kparams.sigma^2*kparams.c_TV;% initial regularization
end

%% PLOW
if(mode == 3)
    kparams.maxval=ScaleMax;
end

%% qGGMRF
if(mode == 4)
    kparams.p=2;
    kparams.q=1.2;
    kparams.c=1e-2;
    kparams.sigmax=0.29;
    kparams.niter = 10; % CHANGE THIS TO AVOID CONFLICT
    kparams.verbose = 0;
    display('Bad variable names for num iter - denoising!');
end
%% General Segmentation parameter
if(mode ==5)
    kparams.num_class=6;
    
    %% MAP Segmentation
    kparams.max_iter=10;
    kparams.filter = params.filter;
    kparams.beta = 3000;
    kparams.debug = 0;
    kparams.sigma_sq = (1/(params.lambda));
    kparams.rand_ord = 0;%Regular ICM or random order ICM
end

%% Augmented Lagrangian Iterations

InitialEst=iradon(sino.counts',sino.theta_0*(180/pi):sino.delta_theta*(180/pi):(sino.n_theta-1)*sino.delta_theta*(180/pi)+sino.theta_0*(180/pi),'cubic','Hamming',sino.n_t);
InitialEst(InitialEst<0)=0;
InitialEst = imresize(InitialEst,[geom.n_y geom.n_x]).*(sino.n_t/geom.n_x);

params.v=InitialEst;
start=tic;
[map_image,params]=ADMM_Core(InitialEst,sino,geom,Amatrix,params,kparams,mode);
elapsed = toc(start);


display(elapsed);

RMSE = sqrt(sum(sum((map_image - Object).^2))/numel(Object));
imagesc(map_image);axis image;colormap(gray);

if(Display == 1)    
    figure;
    imagesc(InitialEst,[ScaleMin ScaleMax]);axis image;colormap(gray)
    colorbar('Eastoutside');
    format_image_for_publication(gcf);
    title('Initial FBP recon');
    
    figure;
    imagesc(map_image,[ScaleMin ScaleMax]);axis image;colormap(gray)
    colorbar('Eastoutside');
    format_image_for_publication(gcf);
    title(strcat('RMSE = ',num2str(RMSE),' \beta = ',num2str(params.beta)));
    
    figure;
    imagesc(params.v,[ScaleMin ScaleMax]);axis image;colormap(gray)
    colorbar('Eastoutside');
    format_image_for_publication(gcf);
    title('v');
    
    if(params.verbose)
        figure;
        plot(params.RMSE);format_plot_for_publication(gcf);
        xlabel('Iteration number');ylabel('RMSE');
    end
    
end
