function [mu_new var_new] = UpdateSegClassParams(mu_old,var_old,Label,ClassLabel,DataVal)
%This function updates the means and variances of the classes when a single
%pixel has been reassigned a class 
%Inputs: mu_old : a 1 X K vector containing the current class means 
%        variance_old : a 1 X K vector containing the class variances
%        Label : an 1 X 2 array with Label(1) being the old value of the 
%                class label and Label(2) being its new value
%        DataVal : The value being moved out from Label(1) to Label(2) 
%        ClassLabel : Current class label map
%        InpImg : Noisy input image being segmented
%Outputs: mu_new : Updated class mean vector
%         var_new : Update class variance vector

M = numel(ClassLabel(ClassLabel == Label(1)));
N = numel(ClassLabel(ClassLabel == Label(2)));
mu_new = mu_old;
var_new = var_old; 
mu_new(Label(1)) = (mu_old(Label(1))*M - DataVal)/(M-1);
mu_new(Label(2)) = (mu_old(Label(2))*N + DataVal)/(N+1);

var_new(Label(1)) = (M*var_old(Label(1)) - (DataVal-mu_old(Label(1)))^2)/(M-1) - (mu_old(Label(1))-mu_new(Label(1)))^2;
var_new(Label(2)) = (N*var_old(Label(2)))/(N+1) + ((DataVal-mu_old(Label(2)))^2)/(N+1);

