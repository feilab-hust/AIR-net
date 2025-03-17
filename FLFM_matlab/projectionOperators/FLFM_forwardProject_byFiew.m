% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function Projection = FLFM_forwardProject_byFiew(H, realSpace,downsampling_size)
% forwardProjectFLFM: Forward projects a volume to a lenslet image by applying the LFPSF
% out=zeros([66,66,477]);

% if downsampling_size~=size(realSpace,1)
%     realSpace=imresize(realSpace,[downsampling_size,downsampling_size],'bicubic');
% end
% out(:,:,166:310)=realSpace(:,:,333:477);
% write3d(out,'./data/GT/volume2_spacing_pixels=66.tif',32)
% write3d(realSpace,'./data/GT/volume2_spacing_pixels=66.tif',32)

Projection = zeros([size(realSpace,1), size(realSpace,2)]);
parfor j = 1:size(H,3)
%     fprintf('[%d/%d] projection\n',j,size(H,3))
    Projection = Projection + conv2(realSpace(:,:,j),H(:,:,j),'same');
end
Projection=imresize(Projection,[downsampling_size,downsampling_size],'bicubic');