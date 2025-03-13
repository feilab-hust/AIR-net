% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function recon = deconvRL_DAO(forwardFUN, backwardFUN, img, iter, init, H, Ht,img_name,save_Path,DAO,Nb)
% Richardson-Lucy deconvolution algorithm
fprintf('\nDeconvolution:')

% Initialize volume
recon = init;
view_num = 9;

for i = 1:iter
    tic
    if DAO>0 % DAO on
        sidelobe=round(0.04*size(img,1)); %% reserved image border
        map_wavshape=zeros(view_num,Nb,Nb,2);     
        Sb=fix( 0.9*size(img,1)/(Nb)/2 )*2+1;
        if Sb<50
            error('Pixel number of single block for multi-AO is too small.');
        end
        border=ceil((size(img,1)-Sb*Nb)/2);
        %weight_mask=single(im2bw(gather(weight),1e-5));
        Sx=(-fix(view_num/2):fix(view_num/2))';
        Sy=Sx;
        
        if i>1
            for view = 1:view_num
                H_temp = H(:,:,view,:);
                fpj = forwardFUN(H_temp,recon);
                
                for uu=1:Nb
                        for vv=1:Nb
                            % tiled correlation with partitioning
                            sub_fpj=fpj(border+(uu-1)*Sb+1:border+uu*Sb,border+(vv-1)*Sb+1:border+vv*Sb);
                            sub_img=img(border+(uu-1)*Sb+1-sidelobe:border+uu*Sb+sidelobe,border+(vv-1)*Sb+1-sidelobe:border+vv*Sb+sidelobe,view);
                            corr_map=gather(normxcorr2(sub_fpj,sub_img));  %每个view进行像差估计（通过比较该view与测量LF的差异计算）
                            [testa,testb]=find(corr_map==max(corr_map(:)));
                            map_wavshape(view,uu,vv,1)=testa-size(sub_img,1)+sidelobe;
                            map_wavshape(view,uu,vv,2)=testb-size(sub_img,2)+sidelobe;       
                        end
                 end
%                  for uu=1:Nb
%                         for vv=1:Nb
%                             cx=map_wavshape(7,7,uu,vv,1);
%                             cy=map_wavshape(7,7,uu,vv,2);
%                             map_wavshape(:,:,uu,vv,1)=(squeeze(map_wavshape(:,:,uu,vv,1))-cx).*weight_mask;   %权重分配
%                             map_wavshape(:,:,uu,vv,2)=(squeeze(map_wavshape(:,:,uu,vv,2))-cy).*weight_mask;
%                         end
%                  end
                  for uu=1:Nb
                        for vv=1:Nb
                            for view_temp = 1:view_num
                                map_wavshape(view_temp,uu,vv,1)=min(max(map_wavshape(view_temp,uu,vv,1),-sidelobe),sidelobe);%% limit the shifted range
                                map_wavshape(view_temp,uu,vv,2)=min(max(map_wavshape(view_temp,uu,vv,2),-sidelobe),sidelobe);%控制移位边缘
                            end
                        end
                  end
                  
                  for uu=1:Nb
                        for vv=1:Nb
                            k1 = Sy.*squeeze(map_wavshape(:,uu,vv,1))+Sx.*squeeze(map_wavshape(:,uu,vv,2));
                            k2 = Sx.*Sx+Sy.*Sy;
                            k=sum(k1(:))/sum(k2(:));
                            map_wavshape(:,uu,vv,1)=squeeze(map_wavshape(:,uu,vv,1))-k*Sy;
                            map_wavshape(:,uu,vv,2)=squeeze(map_wavshape(:,uu,vv,2))-k*Sx;
                            for view_temp = 1:view_num
                                    map_wavshape(view_temp,uu,vv,1)=min(max(map_wavshape(view_temp,uu,vv,1),-sidelobe),sidelobe);
                                    map_wavshape(view_temp,uu,vv,2)=min(max(map_wavshape(view_temp,uu,vv,2),-sidelobe),sidelobe);
                            end
                        end
                  end
            end
        end
    end
    
    % Volume update
    bpjError = zeros(size(recon));
    for view = 1:view_num         %遍历169个view依次迭代重建 
            if DAO>0 && i>1
                % correct aberration  采用上述估计的像差进行像差校正的逐个view重建
                map_wavshape_x=squeeze(map_wavshape(view,:,:,1));
                map_wavshape_y=squeeze(map_wavshape(view,:,:,2));
%                 map_wavshape_x1=imresize(map_wavshape_x,[round(size(WDF,1)/3),round(size(WDF,2)/3)],'nearest');
%                 map_wavshape_y1=imresize(map_wavshape_y,[round(size(WDF,1)/3),round(size(WDF,2)/3)],'nearest');
                map_wavshape_xx=imresize(map_wavshape_x,[size(img,1),size(img,2)],'cubic');
                map_wavshape_yy=imresize(map_wavshape_y,[size(img,1),size(img,2)],'cubic');
                [coordinate1,coordinate2]=meshgrid(1:size(img,1),1:size(img,2));
%                 if GPUcompute==1
%                     img_temp=gpuArray(interp2(coordinate1,coordinate2,img(:,:,view),coordinate1+map_wavshape_yy,coordinate2+map_wavshape_xx,'cubic',0));
%                 else           
                    img_temp=interp2(coordinate1,coordinate2,img(:,:,view),coordinate1+map_wavshape_yy,coordinate2+map_wavshape_xx,'cubic',0); %meshgrid格式的二维插值
%                 end          %从第二次迭代开始，每次迭代遍历所有view，对WDF进行像差校正，之后进行重建
            else
                img_temp=img(:,:,view);
            end
            H_temp = H(:,:,view,:);
            fpj = forwardFUN(H_temp,recon);
            Ht_temp = Ht(:,:,view,:);
            % compute error towards the real image
            errorBack = img_temp./fpj;
            
            % make sure the computations are safe
            errorBack(isnan(errorBack)) = 0;
            errorBack(isinf(errorBack)) = 0;
            errorBack(errorBack < 0) = 0;
            
            % backproject the error
            bpjError = bpjError+backwardFUN(Ht_temp,errorBack);
            
            % update the result
    end
    recon = recon.*bpjError;
    ttime = toc;
    fprintf(['\niter ' num2str(i) ' | ' num2str(iter) ', took ' num2str(ttime) ' secs']);
    if mod(i,5)==0 && i~=0
        recon = gather(recon);
        str_save=fullfile(save_Path,sprintf('deconv_iter%d_DAO_%d.tif',i,DAO));
        
        write3d(recon,str_save,32)
    end
end