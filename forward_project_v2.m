function [sinonew]=forward_project_v2(img,sino,Amatrix)
[m n]=size(img);
sinogram=zeros(sino.n_theta*sino.n_t,1);
for i=1:m
    for j=1:n
%        L=length(Amatrix(i,j).index);
        sinogram(Amatrix(i,j).index)=sinogram(Amatrix(i,j).index)+Amatrix(i,j).value'.*img(i,j);
%        for k=1:L
%            sinogram(Amatrix(i,j).index(k))=sinogram(Amatrix(i,j).index(k))+Amatrix(i,j).value(k)*img(i,j);
%        end
    end
end

for i=1:sino.n_theta
    sinonew(i,:)=sinogram((i-1)*sino.n_t + 1 : i*sino.n_t);
end