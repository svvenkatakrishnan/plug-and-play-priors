function y = downsample2(x, d)
% DOWMSAMPLE2 [y = downsample2(x, d)]
% 2 dimensional downsampling function
% x : input to be downsampled
% d : downsampling factor
%
% coded by hiro on Nov 21, 2004

y = downsample(downsample(x.', d).', d);