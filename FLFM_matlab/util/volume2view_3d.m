function ViewStack=volume2view_3d(volume,pitch,view_num)
img_size=size(volume,1);
depth=size(volume,3);
d=pitch;
d1=0;
MLcenters=zeros(img_size,img_size);
p_center=ceil(d/2);
r=floor(d/2);
center=ceil((img_size)/2);
if view_num==3

    MLcenters(center-ceil((d+d1)/2/sqrt(3)),center-d/2)=1;
    MLcenters(center-ceil((d+d1)/2/sqrt(3)),center+d/2)=1;
    MLcenters(center+ceil((d+d1)/sqrt(3)),center)=1;

end
if view_num==7
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;

    MLcenters(center,center-d-d1)=1;
    MLcenters(center,center)=1;
    MLcenters(center,center+d+d1)=1;

    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;
end
if view_num==19
    MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center-d-d1)=1;
    MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center)=1;
    MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center+d+d1)=1;


    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil(3*(d+d1)/2))=1;
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil(3*(d+d1)/2))=1;
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;


    MLcenters(center,center-2*d-2*d1)=1;
    MLcenters(center,center-d-d1)=1;
    MLcenters(center,center)=1;
    MLcenters(center,center+d+d1)=1;
    MLcenters(center,center+2*d+2*d1)=1;






    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;
    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil(3*(d+d1)/2))=1;
    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil(3*(d+d1)/2))=1;




    MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2),center-d-d1)=1;
    MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2),center)=1;
    MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2),center+d+d1)=1;
end

if view_num==37
    MLcenters(center-ceil(3*sqrt(3)*(d+d1)/2)-1,center-ceil(3*d/2)-d1)=1;
    MLcenters(center-ceil(3*sqrt(3)*(d+d1)/2)-1,center-ceil(d/2))=1;
    MLcenters(center-ceil(3*sqrt(3)*(d+d1)/2)-1,center+ceil(d/2))=1;
    MLcenters(center-ceil(3*sqrt(3)*(d+d1)/2)-1,center+ceil(3*d/2)+d1)=1;

    MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center-2*d-d1)=1;
    MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center-d-d1)=1;
    MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center)=1;
    MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center+d+d1)=1;
    MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center+2*d-d1)=1;

    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil(5*(d+d1)/2))=1;
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil(3*(d+d1)/2))=1;
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil(3*(d+d1)/2))=1;
    MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil(5*(d+d1)/2))=1;

    MLcenters(center,center-3*d-2*d1)=1;
    MLcenters(center,center-2*d-2*d1)=1;
    MLcenters(center,center-d-d1)=1;
    MLcenters(center,center)=1;
    MLcenters(center,center+d+d1)=1;
    MLcenters(center,center+2*d+2*d1)=1;
    MLcenters(center,center+3*d+2*d1)=1;






    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil(5*(d+d1)/2))=1;
    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil(3*(d+d1)/2))=1;
    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;
    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil(3*(d+d1)/2))=1;
    MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil(5*(d+d1)/2))=1;



    MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2)-1,center-2*d-d1)=1;
    MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2)-1,center-d-d1)=1;
    MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2)-1,center)=1;
    MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2)-1,center+d+d1)=1;
    MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2)-1,center+2*d-d1)=1;

    MLcenters(center+ceil(3*sqrt(3)*(d+d1)/2)-1,center-ceil(3*d/2)-d1)=1;
    MLcenters(center+ceil(3*sqrt(3)*(d+d1)/2)-1,center-ceil(d/2))=1;
    MLcenters(center+ceil(3*sqrt(3)*(d+d1)/2)-1,center+ceil(d/2))=1;
    MLcenters(center+ceil(3*sqrt(3)*(d+d1)/2)-1,center+ceil(3*d/2)+d1)=1;
end

[coordiy,coordix]=find(MLcenters==1);

% [coordiy,coordix]=find(MLcenters==1);
coordi=zeros(length(coordiy),2);
for i=1:length(coordiy)
    coordi(i,1) =coordiy(i) ;
    coordi(i,2) =coordix(i) ;
end

[rank_co,ii]=sort(coordi(:,1));
temp=rank_co;
temp(:,2)=coordi(ii,2);
coordi=temp;
mask=@(xx,yy) xx.^2+yy.^2<=r^2;
[yy,xx]=meshgrid(1:pitch,1:pitch);
if (depth~=0)
    ViewStack=zeros(pitch,pitch,depth,view_num);
    
    for ii=1:length(coordi)
        temp_view=zeros(pitch,pitch,depth);
        temp_slice=volume;
        
        hy=coordi(ii,1);
        wx=coordi(ii,2);
        
        
        
        [iy,ix]=find(mask((yy-p_center),(xx-p_center))==1);
         for jj=1:length(iy)
              temp_view(iy(jj),ix(jj),:)=temp_slice(iy(jj)-p_center+hy,ix(jj)-p_center+wx,:);
         end
         ViewStack(:,:,:,ii)=temp_view;
    
     end


else
        ViewStack=zeros(pitch,pitch,view_num);
    
        for ii=1:length(coordi)
            temp_view=zeros(pitch,pitch);
            temp_slice=volume;
        
            hy=coordi(ii,1);
            wx=coordi(ii,2);
        
        
        
            [iy,ix]=find(mask((yy-p_center),(xx-p_center))==1);
            for jj=1:length(iy)
                temp_view(iy(jj),ix(jj))=temp_slice(iy(jj)-p_center+hy,ix(jj)-p_center+wx);
            end
            ViewStack(:,:,ii)=temp_view;
    
           
        end
end
end








