%img = double(imread('~/phd/images/orig/lena.png'));

sigma = 25;


[h w] = size(img);
out = zeros(h,w,5);
mse_out = zeros(1,5);


for mc = 1:5
    y = addWGN(img,sigma,mc-1);
    
    z = plowFast(img, y, sigma);
    
    out(:,:,mc) = z(:,:,end);
    
    mse_out(mc) = mean2((img - z(:,:,2)).^2);
    
end

display('Done');

%save('results/lena_plow_K15_N10_h2sigd_mc.mat','out','mse_out');
    
