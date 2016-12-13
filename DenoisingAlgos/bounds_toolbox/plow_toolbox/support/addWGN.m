function [y vari_n] = addWGN(img,sigma, seed, snr);
% function to add WGN to image

img = double(img);
vari = var(img(:));

if(exist('snr'))
    sigma = 255*(10^(-sigma/20));
end

 if(~exist('seed'))
     seed = 0;
     
 end
randn('state', seed);

% add noisevari_n = vari*10^(-snrout/20);
 % set the noise realization
%y = round0_255(img + randn(size(img)) * sigma);
y = img + randn(size(img)) * sigma;
y(y>255) = 255.0;
y(y<0) = 0.0;
vari_n = var(y(:));
% "y" is the noisy image to be denoised.