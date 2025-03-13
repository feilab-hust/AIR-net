% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu 

function MLARRAY = FLFM_mlaTransmittance(Resolution, ulensPattern)

%% Compute the ML array as a grid of phase/amplitude masks corresponding to mlens
ylength = length(Resolution.yspace);
xlength = length(Resolution.xspace);

% LensletCenters(:,:,1) = round(Resolution.LensletCenters(:,:,2));%[110,324,538,752,966] size=5x5
% LensletCenters(:,:,2) = round(Resolution.LensletCenters(:,:,1));

% activate lenslet centers -> set to 1

MLcenters = zeros(ylength, xlength);
%% rect
% d=floor(Resolution.spacingPixels)*Resolution.superResFactor;
% if mod(d,2)==0
%     d=d-1;
% end
% d1=ceil(0/6.5);
% center=(641*Resolution.superResFactor+1)/2;
% fprintf('\n %d size  -- center %d',d,center);
% MLcenters(center-floor(3/2*d),center-d)=1;
% MLcenters(center-floor(3/2*d),center)=1;
% MLcenters(center-floor(3/2*d),center+d)=1;
% 
% MLcenters(center-floor(1/2*d),center-d)=1;
% MLcenters(center-floor(1/2*d),center)=1;
% MLcenters(center-floor(1/2*d),center+d)=1;
% 
% MLcenters(center+floor(1/2*d),center-d)=1;
% MLcenters(center+floor(1/2*d),center)=1;
% MLcenters(center+floor(1/2*d),center+d)=1;
% 
% MLcenters(center+floor(3/2*d),center-d)=1;
% MLcenters(center+floor(3/2*d),center)=1;
% MLcenters(center+floor(3/2*d),center+d)=1;

for i =1:Resolution.view_num
    MLcenters(Resolution.coordi(i,1),Resolution.coordi(i,2))=1;
end


% apply the mlens pattern at every ml center
MLARRAY  = conv2(MLcenters, ulensPattern, 'same');

