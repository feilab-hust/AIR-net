function ViewStack=FP2view_stacks(fp,pitch,view_num,scale,coordi)
if iscell(fp)
    img_size1=size(fp{1},1);
    img_size2=size(fp{1},2);
    fp1=zeros(img_size1,img_size2,size(fp,3));
    for i=1:size(fp,3)
        fp1(:,:,i)=fp{i};
    end
    fp=fp1;
    
else
    img_size1=size(fp,1);
    img_size2=size(fp,2);
end
d=pitch;
p_center=ceil(d/2);
r=floor(d/2);
mask=@(xx,yy) xx.^2+yy.^2<=r^2;
[yy,xx]=meshgrid(1:pitch,1:pitch);
if (length(size(fp))==3)
    depth=size(fp,3);
    ViewStack=zeros(scale,scale,depth,view_num);
    
    for ii=1:length(coordi)
        temp_view=zeros(pitch,pitch,depth);
        temp_slice=fp;
        
        hy=coordi(ii,1);
        wx=coordi(ii,2);
        
        
        
        [iy,ix]=find(mask((yy-p_center),(xx-p_center))==1);
         for jj=1:length(iy)
              temp_view(iy(jj),ix(jj),:)=temp_slice(iy(jj)-p_center+hy,ix(jj)-p_center+wx,:);
         end
         temp_view=imresize(temp_view,[scale,scale]);
         ViewStack(:,:,:,ii)=temp_view;
    
%          write3d(ViewStack(:,:,:,ii),fullfile(save_Path,sprintf('view_%d_%s',ii,img_name)),bitDepth);
     end


else
        ViewStack=zeros(scale,scale,view_num);
    
        for ii=1:length(coordi)
            temp_view=zeros(pitch,pitch);
            temp_slice=fp;
        
            hy=coordi(ii,1);
            wx=coordi(ii,2);
        
        
        
            [iy,ix]=find(mask((yy-p_center),(xx-p_center))==1);
            for jj=1:length(iy)
                temp_view(iy(jj),ix(jj))=temp_slice(iy(jj)-p_center+hy,ix(jj)-p_center+wx);
            end
            temp_view=imresize(temp_view,[scale,scale]);
            ViewStack(:,:,ii)=temp_view;
            
    
           
        end
    end
end








