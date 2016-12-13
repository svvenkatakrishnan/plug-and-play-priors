function [z, zx1, zx2] = ckr2reg(y, h, ksize)
% [CKR2_REGULAR]
% The second order classic kernel regression function for regularly sampled
% data.
%
% [USAGE]
% [z, zx1, zx2] = ckr2_regular(y, h, r, ksize)
%
% [RETURNS]
% z     : the estimated image
% zx1   : the estimated gradient image along the x1 direction (vertical
%        direction)
% zx2   : the estimated gradient image along the x2 direction (horizontal
%        direction)
%
% [PARAMETERS]
% y     : the input image
% h     : the global smoothing parameter
% r     : the upscaling factor ("r" must be an integer number)
% ksize : the size of the kernel (ksize x ksize, and "ksize" must be
%         an odd number)
%
% [HISTORY]
% June 16, 2007 : created by Hiro

% Get the oritinal image size
[N, M] = size(y);

% Initialize the return parameters
z = zeros(N, M);
zx1 = zeros(N, M);
zx2 = zeros(N, M);

% Create the equivalent kernels
radius = (ksize - 1) / 2;
[xx2, xx1] = meshgrid(-radius:radius, -radius:radius);
A = zeros(3, ksize^2);

% The feture matrix
Xx = [ones(ksize^2,1), xx1(:), xx2(:)];
        %Xx = [ones(ksize^2,1), xx1(:), xx2(:), xx1(:).^2, xx1(:).*xx2(:), xx2(:).^2];
% The weight matrix (Gaussian kernel function)
tt = xx1.^2 + xx2.^2;
W = exp(-(0.5/h^2) * tt);
% Equivalent kernel
Xw = Xx.*repmat(W(:),[1 size(Xx,2)]);
%Xw = [Xx(:,1).*W(:), Xx(:,2).*W(:), Xx(:,3).*W(:),...
%     Xx(:,4).*W(:), Xx(:,5).*W(:), Xx(:,6).*W(:)];
A = inv(Xx.' * Xw) * (Xw.');

% Mirroring the input image
y = padarray(y, [radius, radius], 'symmetric');

% Estimate an image and its first gradients with pixel-by-pixel
for n = 1 : N
    for m = 1 : M
        
        % Neighboring samples to be taken account into the estimation
        yp = y(n:n+ksize-1, m:m+ksize-1);
        
        % Estimate the pixel values at (nn,mm)
        z(n,m)   = A(1,:) * yp(:);
        zx1(n,m) = A(2,:) * yp(:);
        zx2(n,m) = A(3,:) * yp(:);
        
    end
end
