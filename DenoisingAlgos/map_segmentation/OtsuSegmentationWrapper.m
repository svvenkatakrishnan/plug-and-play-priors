function [Recon]=OtsuSegmentationWrapper(Y,N)
%function which takes a noisy image Y and segments it into "N" classes
%where N is stored in using Otsu's method. The image is then reconstructed
%using the mean value of the class
%Inputs : Y : input image
%         N : Number of segmentation classes
%Outputs : Recon : each pixel replaced by the mean values of its class
Recon = zeros(size(Y));
[IDX,sep] = otsu(Y,N);
[Recon,ClassMean,ClassVar]=ClassLabel2GrayScale(Y,IDX,N);