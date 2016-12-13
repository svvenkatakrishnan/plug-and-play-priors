function W = getLARK(img, wsize, h)


[N,M] = size(img);
ksize = 11;
win = (ksize-1)/2;
img1 = padarray(img,[win win],'symmetric');
% % [zx,zy] = gradient(img1);
[~, zx, zy] = ckr2reg(img1, 0.5,5); 
%img1 = conv2(img1, fspecial('gaussian', [win win], 0.5), 'same');
%zx = conv2(img1, [0 0 0; -1 0 1; 0 0 0], 'same'); 
%zy = conv2(img1, [0 -1 0; 0 0 0; 0 1 0], 'same');
clear img1;

K = fspecial('disk', win);
K = K ./ K(win+1, win+1);
% figure, imagesc(uint8(zx)), colormap(gray), axis image;
% figure, imagesc(uint8(zy)), colormap(gray), axis image;
len = sum(K(:));
lambda = 1;
alpha = .5;

len = 121; 

SSS = zeros(N,M);
C11 = zeros(N,M);
C12 = C11; C22 = C11;

for j = 1 : M
    for i = 1 : N
        %gx = zx(i:i+ksize-1, j:j+ksize-1).* K;
        %gy = zy(i:i+ksize-1, j:j+ksize-1).* K;
        gx = zx(i:i+ksize-1, j:j+ksize-1);
        gy = zy(i:i+ksize-1, j:j+ksize-1);
        G = [gx(:), gy(:)];
        
        [~, s v] = svd(G);
        
        
        S(1) = (s(1,1) + lambda) / (s(2,2) + lambda);
        S(2) = (s(2,2) + lambda) / (s(1,1) + lambda);
        
        SSS(i,j) = v(2,1);
        
        tmp = (S(1) * v(:,1) * v(:,1).' + S(2) * v(:,2) * v(:,2).')  * ((s(1,1) * s(2,2) + 0.0000001) / len)^alpha;
        C11(i,j) = tmp(1,1);
        C12(i,j) = tmp(1,2);
        C22(i,j) = tmp(2,2);
       % sq_detC(i,j) = sqrt(det(tmp));
    end
end


win = (wsize-1)/2;
[x2,x1] = meshgrid(-win:win,-win:win);
C11 = padarray(C11,[win win],'symmetric');
C12 = padarray(C12,[win win],'symmetric');
C22 = padarray(C22,[win win],'symmetric');
%sq_detC = padarray(sq_detC,[win win],'symmetric');

%figure; imagesc(SSS); axis image; colorbar;
% figure; imagesc(C22); axis image; colorbar;
% figure; imagesc(C12); axis image; colorbar;

W = zeros(wsize^2, N*M);

h2 = -0.5/(h^2);
wsz = wsize^2;
k=1;

for m = 1 : M
    for n = 1 : N
        tt = x1 .* (C11(n:n+wsize-1, m:m+wsize-1) .* x1...
            + C12(n:n+wsize-1, m:m+wsize-1) .* x2)...
            + x2 .* (C12(n:n+wsize-1, m:m+wsize-1) .* x1...
            + C22(n:n+wsize-1, m:m+wsize-1) .* x2);
        %W(:,k) = reshape(exp(-(0.5/h2) * tt).* sq_detC(n:n+wsize-1, m:m+wsize-1),[wsize^2 1]);
        W(:,k) = exp(h2 * tt(:));
        k = k+1;
    end
end
