% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function [H, Ht] = FLFM_computeLFMatrixOperators(Camera, Resolution,ax2ca,mla2axicon,is_axicon)

%% Sensor and ML space coordinates
IMGsizeHalf_y = floor(Resolution.sensorSize(1)/2);
IMGsizeHalf_x = floor(Resolution.sensorSize(2)/2);
Resolution.yspace = Resolution.sensorRes(1)*linspace(-IMGsizeHalf_y, IMGsizeHalf_y, Resolution.sensorSize(1));  %sensor plane coordinates
Resolution.xspace = Resolution.sensorRes(2)*linspace(-IMGsizeHalf_x, IMGsizeHalf_x, Resolution.sensorSize(2));
Resolution.yMLspace = Resolution.sensorRes(1)* [- ceil(Resolution.Nnum(1)/2) + 1 : 1 : ceil(Resolution.Nnum(1)/2) - 1];   %local lenslet coordinates
Resolution.xMLspace = Resolution.sensorRes(2)* [- ceil(Resolution.Nnum(2)/2) + 1 : 1 : ceil(Resolution.Nnum(2)/2) - 1];

%% Compute LFPSF operators

% compute the wavefront distribution incident on the MLA for every depth
fprintf('\nCompute the PSF stack at the back aperture stop of the MO.')
psfSTACK = FLFM_calcPSFAllDepths(Camera, Resolution);

% compute LFPSF at the sensor plane
fprintf('\nCompute the LFPSF stack at the camera plane:')
tolLFpsf = 0.001; % clap small valueds in the psf to speed up computations

for ii=1:length(ax2ca)
    d_camera=ax2ca(ii);
    [H, Ht] = FLFM_computeLFPSF(psfSTACK, Camera, Resolution, tolLFpsf,d_camera,mla2axicon,is_axicon);
    HS=psf_process_func(H, 99,Resolution);
    save('./psf_matrix/HS.mat',"HS",'-v7.3');
    for jj = 1:Resolution.view_num
       write3d(permute(squeeze(HS(jj,:,:,:)),[2,3,1]),fullfile('./PSF_MATRIX/psf_imgs',sprintf('psf_raw_%d.tif',jj)),32);
    end
   
end
