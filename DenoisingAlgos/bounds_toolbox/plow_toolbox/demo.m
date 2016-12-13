%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% The following is a demo code that illustrates the different options     % 
% that can be used to run PLOW code.                                      %
%                                                                         %
% If you are using this code to generate results, please cite the         % 
% following papers:                                                       %
%                                                                         %
% P. Chatterjee, P. Milanfar, "Patch-based Near-Optimal Image Denoising", %
% IEEE Transactions on Image Processing, to appear, 2012.                 %
%                                                                         %
% P. Chatterjee, and P. Milanfar, "Patch-based Locally Optimal Denoising",%
% Proc. of IEEE Intl. Conf. on Image Processing, pp. 2553-2556, Sept.     %
% 2011.                                                                   %
%                                                                         %
% For any queries regarding this code, please contact priyam@soe.ucsc.edu %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('support');

img = phantom(64).*255;%double(imread('image/house.png'));
img(img<0)=0;

sigma = 15;

% Form noisy image
y = addWGN(img,sigma,0);

%% Run PLOW, with an initial estimate of the noise

% In some high noise cases, for some images, better denoising performance
% may be obtained by using a lower than actual noise estimate.

%z = plowFast(img, y, sigma);

%% Run PLOW without any estimate of the noise

% z = plowFast(img, y);

%% Run PLOW, but a faster version compromising a little on PSNR

% skip = 3;
% z = plowFast(img, y, sigma, skip);

%% If the previous versions exit with Out of Memory error, try this slower
% but less memory consuming version

% z = plow(img, y, sigma);

%% if you cannot run the Mex versions, try this -- beware, can be slow!!
 z = plowMatlab(img, y, sigma);

%% Compute PSNR and display images

mOut = mean2((img - z(:,:,end)).^2);
pOut = 10*log10((255^2)/mOut);
display(strcat('Output PSNR is :', num2str(pOut)));

figure; imagesc(uint8(y)); axis image; colormap gray; caxis([0 255]); title('Noisy image');
figure; imagesc(uint8(z(:,:,end))); axis image; colormap gray; caxis([0 255]); title('Denoised image');

