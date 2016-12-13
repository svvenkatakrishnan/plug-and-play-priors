function Y = RGB2YCC(seq)

[N M c num] = size(seq);
Y  = zeros(N, M, num);
Cr = zeros(N, M, num);
Cb = zeros(N, M, num);

frames = double(seq);
for i = 1 : num
    Y(:,:,1)  =  0.299 * frames(:,:,1,i) + 0.587 * frames(:,:,2,i) + 0.114 * frames(:,:,3,i);
    Y(:,:,3) = -0.168736 * frames(:,:,1,i) - 0.331264 * frames(:,:,2,i) + 0.5 * frames(:,:,3,i);
    Y(:,:,2) =  0.5 * frames(:,:,1,i) - 0.418668 * frames(:,:,2,i) - 0.081312 * frames(:,:,3,i);
end