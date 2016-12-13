function u = Solve(x,theta1,theta2,sigmax,p,q,c,location)
% function u = Solve(x,theta1,theta2,sigmax,p,q,c,location)
% This function computes the updated voxel value.
% Input:
%   x: image
%   theta1: first derivative of the log likelihood w.r.t. the chosen voxel
%   theta2: second derivative of the log likelihood w.r.t. the chosen voxel
%   sigmax: regularization parameter
%   p, q, c: parameters for the q-GGMRF prior
%   location: [i,j], the coordinate of the chosen voxel
% Output:
%   u: updated voxel value
%

g1 = 1/6; g2 = 1/12; % weights for local voxel pair

[m n]=size(x);

% compute surrogate prior parameter phi1 and phi2
i = location(1); j = location(2);
xs = x(i,j);

% compute the coefficients of the log likelihood term
phi1 = theta1 - theta2*xs;
phi2 = theta2;

% compute the coefficient for surrogate prior for each neighboring voxel
if j-1>0
    xr = x(i,j-1);
    vd = xs-xr;
    if vd~=0
        psi = Compute_Surrogate_Prior(vd,sigmax,p,q,c);
    else
        tmp1 = (Compute_Surrogate_Prior(0.001,sigmax,p,q,c))*.001;
        tmp2 = (Compute_Surrogate_Prior(-0.001,sigmax,p,q,c))*(-.001);
        psi = (tmp1-tmp2)/.002; % approximate the second derivate at origin       
    end
    phi1 = phi1 - g1.*psi*xr;
    phi2 = phi2 + g1.*psi;
end

if j+1<n+1
    xr = x(i,j+1);
    vd = xs-xr;
    if vd~=0
        psi = Compute_Surrogate_Prior(vd,sigmax,p,q,c);
    else
        tmp1 = (Compute_Surrogate_Prior(0.001,sigmax,p,q,c))*.001;
        tmp2 = (Compute_Surrogate_Prior(-0.001,sigmax,p,q,c))*(-.001);
        psi = (tmp1-tmp2)/.002; % approximate the second derivative at origin       
    end
    phi1 = phi1 - g1.*psi*xr;
    phi2 = phi2 + g1.*psi;
end

if i-1>0
    xr = x(i-1,j);
    vd = xs-xr;
    if vd~=0
        psi = Compute_Surrogate_Prior(vd,sigmax,p,q,c);
    else
        tmp1 = (Compute_Surrogate_Prior(0.001,sigmax,p,q,c))*.001;
        tmp2 = (Compute_Surrogate_Prior(-0.001,sigmax,p,q,c))*(-.001);
        psi = (tmp1-tmp2)/.002; % approximate the second derivate at origin       
    end
    phi1 = phi1 - g1.*psi*xr;
    phi2 = phi2 + g1.*psi;

    if j-1>0
        xr = x(i-1,j-1);
        vd = xs-xr;
        if vd~=0
            psi = Compute_Surrogate_Prior(vd,sigmax,p,q,c);
        else
            tmp1 = (Compute_Surrogate_Prior(0.001,sigmax,p,q,c))*.001;
            tmp2 = (Compute_Surrogate_Prior(-0.001,sigmax,p,q,c))*(-.001);
            psi = (tmp1-tmp2)/.002; % approximate the second derivate at origin       
        end
        phi1 = phi1 - g2.*psi*xr;
        phi2 = phi2 + g2.*psi;
   end


    if j+1<n+1
        xr = x(i-1,j+1);
        vd = xs-xr;
        if vd~=0
            psi = Compute_Surrogate_Prior(vd,sigmax,p,q,c);
        else
            tmp1 = (Compute_Surrogate_Prior(0.001,sigmax,p,q,c))*.001;
            tmp2 = (Compute_Surrogate_Prior(-0.001,sigmax,p,q,c))*(-.001);
            psi = (tmp1-tmp2)/.002; % approximate the second derivate at origin       
        end
        phi1 = phi1 - g2.*psi*xr;
        phi2 = phi2 + g2.*psi;
    end
end

if i+1<m+1
    xr = x(i+1,j);
    vd = xs-xr;
    if vd~=0
        psi = Compute_Surrogate_Prior(vd,sigmax,p,q,c);
    else
        tmp1 = (Compute_Surrogate_Prior(0.001,sigmax,p,q,c))*.001;
        tmp2 = (Compute_Surrogate_Prior(-0.001,sigmax,p,q,c))*(-.001);
        psi = (tmp1-tmp2)/.002; % approximate the second derivate at origin       
    end
    phi1 = phi1 - g1.*psi*xr;
    phi2 = phi2 + g1.*psi;

    if j-1>0
        xr = x(i+1,j-1);
        vd = xs-xr;
        if vd~=0
            psi = Compute_Surrogate_Prior(vd,sigmax,p,q,c);
        else
            tmp1 = (Compute_Surrogate_Prior(0.001,sigmax,p,q,c))*.001;
            tmp2 = (Compute_Surrogate_Prior(-0.001,sigmax,p,q,c))*(-.001);
            psi = (tmp1-tmp2)/.002; % approximate the second derivate at origin       
        end
        phi1 = phi1 - g2.*psi*xr;
        phi2 = phi2 + g2.*psi;
   end

    if j+1<n+1
        xr = x(i+1,j+1);
        vd = xs-xr;
        if vd~=0
            psi = Compute_Surrogate_Prior(vd,sigmax,p,q,c);
        else
            tmp1 = (Compute_Surrogate_Prior(0.001,sigmax,p,q,c))*.001;
            tmp2 = (Compute_Surrogate_Prior(-0.001,sigmax,p,q,c))*(-.001);
            psi = (tmp1-tmp2)/.002; % approximate the second derivate at origin       
        end
        phi1 = phi1 - g2.*psi*xr;
        phi2 = phi2 + g2.*psi;
    end
end

% compute the solution
u = max(-phi1/phi2, 0);
