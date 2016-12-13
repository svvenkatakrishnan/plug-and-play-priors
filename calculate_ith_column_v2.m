function [value,index]= calculate_ith_column_v2(row,col,geom,sino,det_resp)

x=geom.x_0 + (col-1)*geom.delta + geom.delta/2;
z=geom.y_0 + (row-1)*geom.delta + geom.delta/2;
count=1;


[m DETECTOR_RESPONSE_BINS]=size(det_resp);

OffsetR = (geom.delta/sqrt(3.0) + sino.delta_t/ 2)/DETECTOR_RESPONSE_BINS;

%%%%Defining the beam profile - Raised Cosine
BeamWidth= sino.delta_t;%/20;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for k=1:sino.n_theta
    
    t=x*sino.cosine(k)- z*sino.sine(k) ;
    t_min = t - geom.delta;
    t_max = t + geom.delta;
    
    if(t_max < sino.t_0)
        continue;
    end
    
    index_min = floor(((t_min - sino.t_0)/sino.delta_t)) ;
    index_max = floor(((t_max - sino.t_0)/sino.delta_t)) ;
    
    if(index_min < 0)
        index_min = 0;
    else if(index_max >= sino.n_t)
            index_max = sino.n_t-1;
        end
    end
    
    base_index_Ai=(k-1)*sino.n_t + 1;%+1 for matlab indexing
    
    for i=index_min:index_max
        
        t_0 = sino.t_0 + (double(i)+0.5)*(sino.delta_t); %detector center - beam is centered here
               
        %Find the difference between the center of detector and center of projection and compute the Index to look up into
        delta_r = abs(t - t_0);
        index_delta_r = (floor((delta_r / OffsetR)));
        
        if(index_delta_r >= 0 && index_delta_r < DETECTOR_RESPONSE_BINS)
            %Using index_delta_r and index_delta_r+1 do bilinear interpolation
            w1 = delta_r - index_delta_r*OffsetR;
            w2 = (index_delta_r + 1)*OffsetR - delta_r;
            
            if(index_delta_r+1 <= DETECTOR_RESPONSE_BINS-1)
                iidx = int16(index_delta_r)+1;
            else
                iidx = int16(DETECTOR_RESPONSE_BINS)-1;
            end
            
            f1 = (w2 / OffsetR) * det_resp(k,index_delta_r+1)+ (w1 / OffsetR) * det_resp(k, iidx+1);
                        
            InterpolatedValue = f1;
            
            %modify
            if(InterpolatedValue > 0)
                index(count) = (base_index_Ai) + i;
                value(count) = InterpolatedValue;
                count= count +1;
            end
        end
    end
end
 if(count ==1)
     display('No entries in column');
     display(row);
     display(col);
 end
