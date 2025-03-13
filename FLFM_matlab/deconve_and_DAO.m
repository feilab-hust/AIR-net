clc
clear
addpath ./util
addpath ./solvers
addpath ./LFPSF
addpath ./projectionOperators
DAO=1;
view_num=9;
time_seq=false;
parallel.gpu.enableCUDAForwardCompatibility(true)
%% loading PSF
load('../code/data/exp_data/HS.mat')
psf_ViewStack=permute(HS(:,:,:,:),[3,4,2,1]);
images=cell(1,view_num);
for i=1:view_num
    images{i}=psf_ViewStack(:,:,:,i);
end
H_size=size(full(images{1}));
H=zeros(H_size(1),H_size(2),view_num,H_size(3));
Ht=zeros(H_size(1),H_size(2),view_num,H_size(3));
for i=1:view_num
    H(:,:,i,:)=images{i};
    for j=1:H_size(3)       
        Ht(:,:,i,j)=imrotate(H(:,:,i,j),180);
    end
end

[hr_file_name,hr_filepath] = uigetfile('*.tif;*.tiff','Select LR_VIEW','MultiSelect','on');
if ~iscell(hr_file_name)
    hr_file_name = {hr_file_name};
end



for img_idx=1:length(hr_file_name)
    img_name=hr_file_name{img_idx};
    fprintf('\\Deconv:%s',img_name);
    img_path=fullfile(hr_filepath,img_name);
    file_name=sprintf('\\Deconv');
    save_Path=fullfile(hr_filepath,file_name);
    if exist(save_Path,'dir')==7
        ;
    else
        mkdir(save_Path);
    end
    ViewStack=imread3d(img_path);
    ViewStack=double(ViewStack);
%     for i=1:view_num
%      ViewStack(:,:,i)=imrotate(ViewStack(:,:,i),180);
%     end
    volumeSize = [size(ViewStack,1), size(ViewStack,2), size(Ht,4)];
    if time_seq
        if mod(img_idx-1,10)==0
            init = ones(volumeSize);
            iter =15; % number of iterations
        end
        if mod(img_idx-1,11)==1 
            init=recon;
            iter =5;
        end
    else
            init = ones(volumeSize);
            iter =15; % number of iterations
    end
    forwardFUN = @(H,volume) FLFM_forwardProjectGPU(H, volume);
    backwardFUN = @(Ht,projection) FLFM_backwardProjectGPU(Ht, projection);
    global zeroImageEx;
    global exsize;
    xsize = [volumeSize(1), volumeSize(2)];
    msize = [H_size(1), H_size(2)];
    mmid = floor(msize/2);
    exsize = xsize + mmid;  
    exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];    
    zeroImageEx = gpuArray(zeros(exsize, 'single'));
%     disp(['FFT size is ' num2str(exsize(1)) 'X' num2str(exsize(2))]); 
    
    %%  Richardson Lucy deconvolution
    tic
    recon = deconvRL_DAO(forwardFUN, backwardFUN, ViewStack, iter, init, H, Ht,img_name,save_Path,DAO,1);;
    ttime = toc;
    fprintf(['img_idx ' num2str(img_idx) ' | ' num2str(length(hr_file_name)) ', took ' num2str(ttime) ' secs']);
end



