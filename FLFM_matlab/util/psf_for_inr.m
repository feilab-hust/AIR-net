addpath('./util');
save_path='./iso_psf/lambda1100';
if exist(save_path, 'dir')==0
    mkdir(save_path);
end
load('./PSF_MATRIX\psf_mat\osr2_light_sheet_dof200.mat');
% profile=imread3d('./iso_psf/PSF BW.tif');
% profile=cut_circ_view(profile,size(profile,1));
use_profile=false;
pitch=303;
view_num=9;
scale=303;
psf_w=99;
volume=zeros(size(H{:,:,1},1),size(H{:,:,1},2),size(H,3));
HS=zeros(view_num,size(H,3),psf_w,psf_w);
for i=1:size(H,3)
volume(:,:,i)=H{:,:,i};
end
%% localization
psf_view=FP2view_stacks(volume,pitch,view_num,scale);

for i=1:view_num
    fprintf('proicessing[%d]/[%d]\n',i,view_num);
%     psf_3d=imresize(psf_view(:,:,:,i),[size(psf_view,1)*OSR,size(psf_view,2)*OSR],'bicubic');
    psf_3d=psf_view(:,:,:,i);
%     psf_3d=imresize(psf_3d,[scale,scale]);
%     psf_3d_new=imresize(psf_3d_new,[pitch*3,pitch*3]);   
    write3d(psf_3d(ceil(scale/2)-floor(psf_w/2):ceil(scale/2)+floor(psf_w/2),ceil(scale/2)-floor(psf_w/2):ceil(scale/2)+floor(psf_w/2),:),fullfile(save_path,sprintf('psf_view%d.tif',i)),32);
    psf_3d=permute(psf_3d,[3,1,2]);
    HS(i,:,:,:)=psf_3d(:,ceil(scale/2)-floor(psf_w/2):ceil(scale/2)+floor(psf_w/2),ceil(scale/2)-floor(psf_w/2):ceil(scale/2)+floor(psf_w/2));
   
   
end
save('HS.mat',"HS");

