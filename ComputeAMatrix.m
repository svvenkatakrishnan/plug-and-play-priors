function [imstruct]=ComputeAMatrix(sino,geom)
m=geom.n_y;%vertical dim
n=geom.n_x;%horizontal dim
sinogram=zeros(sino.n_theta*sino.n_t,1);
pix_profile = calc_pix_profile(sino,geom);
det_resp = calc_det_response(pix_profile,sino,geom);
count=0;
for i=1:m
    count=count+1;
    display(count);
    for j=1:n      
      % [value,index]= calculate_ith_column(i,j,geom,sino,pix_profile);
       % [value,index]= calculate_ith_column_DD(i,j,geom,sino);
       [value,index]= calculate_ith_column_v2(i,j,geom,sino,det_resp);
        imstruct(i,j).value=value;
        imstruct(i,j).index=index;
    end
end