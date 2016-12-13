function [det_resp] = calc_det_response(pix_profile,sino,geom)
%Response of the 2-D detector to the pixel profile as a function of
%displacement from center

BEAM_RESOLUTION = 512;
DETECTOR_RESPONSE_BINS = 64;
PROFILE_RESOLUTION = 512;

beamWidth = sino.delta_t;
offsetR = (geom.delta/sqrt(3.0) + sino.delta_t/ 2)/DETECTOR_RESPONSE_BINS;

beamProfile = 0.54 - 0.46*cos(2*(pi/BEAM_RESOLUTION).*[0:BEAM_RESOLUTION-1]);
beamProfile= beamProfile./sum(beamProfile(:)); %Normalize the kernel used to average the response

r0 = -(beamWidth)/2;
StepSize = beamWidth/BEAM_RESOLUTION;

det_resp=zeros(sino.n_theta,DETECTOR_RESPONSE_BINS);

TempConst=(PROFILE_RESOLUTION)/(2*geom.delta);

for k = 0:sino.n_theta-1
    for i = 0:DETECTOR_RESPONSE_BINS-1 %displacement along r
        ProfileCenterR = i*offsetR; %Where the profile center "lands"
        rmin = ProfileCenterR - geom.delta; %Left corner of the stored profile
        tmp_sum = 0;
        for p = 0:BEAM_RESOLUTION-1 %Apply inner product of beam profile with pixel profile
            
            r = r0 + p * StepSize;
            if(r < rmin)
                continue;
            end
            ProfileIndex = floor((r - rmin) * TempConst);
            ProfileIndex = int16(ProfileIndex);
            if(ProfileIndex < 0)
                ProfileIndex = 0;
            end
            if(ProfileIndex >= PROFILE_RESOLUTION)
                ProfileIndex = PROFILE_RESOLUTION - 1;
            end
            
            tmp_sum = tmp_sum+pix_profile(k+1, ProfileIndex+1) * beamProfile(p+1);
        end
        det_resp(k+1,i+1)=tmp_sum;
    end
end