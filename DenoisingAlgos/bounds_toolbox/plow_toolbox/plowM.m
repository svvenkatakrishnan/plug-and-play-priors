function [Ztmp Ce] = plowM(y2, y, my, V, D, Ridx, Cidx, sig2, ksz, thresh, prepFlag)

d = ksz^2;
csz = numel(Ridx);
p = 10;
Ztmp = zeros(d,csz);
Ce = Ztmp;


ssz = 30;
wrad = ssz/2;

[h w] = size(y2);
h2 = 1.75*sig2*d;

for i=1:csz
    
    r = Ridx(i); c = Cidx(i);

    
    yi = y2( r:r+ksz-1, c:c+ksz-1 );
    yi = yi(:);
    
    
     
    rmin = max(1, r - wrad);
    cmin = max(1,c - wrad);
    rmax = min(h, r + wrad);
    cmax = min(w, c + wrad);
     
    Y = im2col(y2(rmin:rmax,cmin:cmax), [ksz ksz], 'sliding');
    
    [idx_nn p2 nndist2] = getNN2(Y,yi,thresh,p);

    
    if(prepFlag)
        mul = double(p2)/sig2;
    else
        mul = 1.0/sig2;
    end
    wts = mul*exp(-nndist2'./h2); 
    
    wSum = sum(wts);  
    D2 = D.*wSum;
      
    errMat = V*diag(D./(D2+1))*V';
    
    Y = im2col(y(rmin:rmax,cmin:cmax), [ksz ksz], 'sliding');
    
    Ztmp(:,i) = my + errMat*sum((Y(:,idx_nn) - my(:,ones(p2,1)))*wts,2);
    Ce(:,i) = diag(errMat);
end
