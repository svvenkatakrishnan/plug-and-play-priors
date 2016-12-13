function [idx_nn p D] = getNN2(Yk,yi,eps,k)
% Function that returns the p-nearest neighbors such that
% dist(yi, yj) <= eps. Dist is the squared euclidian distance

%ksz = size(yi,1);

%D = (sum((Yk - Yi).^2))';
Yk = bsxfun(@minus, Yk, yi); 
D = sum(Yk.^2);
clear Yk;

[D idx_nn] = sort(D);
p = min(numel(idx_nn),k);
idx_nn = idx_nn(1:p);
D = D(1:p);

lis = find(D<=eps); 
idx_nn = idx_nn(lis);
D = D(lis);
p = numel(lis);

end