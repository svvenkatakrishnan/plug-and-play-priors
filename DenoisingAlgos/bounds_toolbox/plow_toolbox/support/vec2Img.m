function img = vec2Img(Z,R,dims,skip)

if(~exist('skip','var'))
    skip = 1;
end

h = dims(1); w = dims(2);


patchSz = sqrt(size(Z,1));
rad = (patchSz - 1)/2;
img = zeros(h+patchSz-1,w+patchSz-1,'double');

if(skip>1)
    h2 = ceil(h/skip);
    w2 = ceil(w/skip);
    
    % Check for dimension match
    if(size(Z,2) ~= h2*w2)
        display('Error in estimating dimensions: Not all patches may be present');
        display(strcat('Size of data received : ', num2str(size(Z,2))));
        display(strcat('Expected size : ', num2str(h2), ' x ', num2str(w2)));
        return;
    end
    
else
    h2 = h;
    w2 = w;
end


W = img;

Z = Z./R;

idx = 0;

for j=1:patchSz
   
    for i = 1:patchSz
        
        idx = idx+1;
        img(i:skip:i+h-1,j:skip:j+w-1) =  img(i:skip:i+h-1,j:skip:j+w-1) + reshape(Z(idx,:),[h2 w2]);
        W(i:skip:i+h-1,j:skip:j+w-1) =  W(i:skip:i+h-1,j:skip:j+w-1) + 1./reshape(R(idx,:),[h2 w2]);
        
    end
    
end

img = img./W;
img = img(rad+1:h+rad,rad+1:w+rad);