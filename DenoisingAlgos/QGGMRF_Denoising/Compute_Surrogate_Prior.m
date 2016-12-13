function value = Compute_Surrogate_Prior(voxel_distance,sigma,p,q,c)
% function value = Compute_Surrogate_Prior(voxel_distance,sigma,p,q,c)
% This function computes the coefficient for the quadratic term of the surrogate prior.
% Input:
%   voxel_distance: the difference between the chosen voxel and its neighbor
%   sigma: regularization parameter
%   p, q, c: parameters for the q-GGMRF prior
% Output:
%   value: the coefficient for the quadratic term of the surrogate prior
%

vd = abs(voxel_distance);
tmp = p-q;

denom = 1+(vd/c)^tmp;
value = p - (tmp*vd^tmp)/(denom*c^tmp);
value = value*(vd^(p-2))/denom;
value = value/(p*sigma^p);
