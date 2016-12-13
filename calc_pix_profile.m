function [pix_profile]=calc_pix_profile(sino,geom)
resolution = 512;
pix_profile=zeros(sino.n_theta,resolution);
scale=geom.delta;
for i=1:sino.n_theta
    angle=sino.theta_0 + (i-1)*sino.delta_theta;
    
    while(angle > pi/2)
        angle =angle - pi/2;
    end

    while(angle < 0)
         angle =pi/2 + angle;
    end
    
    if(angle <= pi/4)
        max_val_line_integral = scale/cos(angle);
    else
        max_val_line_integral = scale/cos(pi/2-angle);
    end
    
    rc=cos(pi/4);
            
    d1 = rc * cos((pi/4.0-angle));
    d2 = rc*abs((cos((pi/4.0+angle))));
    t_1 = 1-d1;
    t_2 = 1-d2;
    t_3 = 1+d2;
    t_4 = 1+d1;
    for j = 1:resolution
        
        t = 2.0*j / resolution;
        
        if(t<=t_1 || t>=t_4)
            pix_profile(i,j) = 0;
        else if(t>t_3)
                pix_profile(i,j) = max_val_line_integral*(t_4-t)/(t_4-t_3);
            else if(t>=t_2)
                    pix_profile(i,j) = max_val_line_integral;
                else
                    pix_profile(i,j) = max_val_line_integral*(t-t_1)/(t_2-t_1);
                end
            end
        end
    end
end
