function [cost]=Cost_DataTermAL(error,weight,x,params)
% Cost assosiated with the data term optimization in Augmented Lagrangian
% error holds the current error as a vector
% weight is the weight assosiated with the error vector
% x is the current 2-D image
% params is a struct which holds the AL parameters: lambda, v and rho

cost = (error.*weight)*error';
temp = x-params.v+params.u;
cost = cost + params.lambda*sum(sum((temp.*temp)));
cost = cost/2;