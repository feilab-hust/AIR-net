function HS = psf_process_func(H, psf_w,Resolution)
    addpath('./util');  %  
    % 視圖生成 
    psf_view = FP2view_stacks(H, psf_w, Resolution.view_num, psf_w,Resolution.coordi);
    
    % 視圖處理流水線 
    HS = process_views(psf_view, Resolution.view_num, psf_w );
    
end 
 
 
function HS = process_views(psf_view, view_num,psf_w)
    HS = zeros(view_num, size(psf_view,3), psf_w, psf_w);
    
    for i = 1:view_num  
        fprintf('Processing view %d/%d\n', i, view_num);
        view_data = psf_view(:,:,:,i);    
        HS(i,:,:,:) = permute(view_data, [3,1,2]);
    end 
end 