import mat73
import torch
import numpy as np
from core.utils.fft_conv import fft_conv


def forward_project(realspace:torch.Tensor, H_mtx:torch.Tensor, padding:str='same'):
    """
    Perform forward projection.
    realspace: object space volume voxel
    H_mtx: H matrix used for forward projection. (Related to PSF)
    padding: padding added to each dim and each side. See torch.nn.functional.pad().
    """
    D,H,W,C = realspace.shape
    Dm,Hm,Wm = H_mtx.shape
    assert D == Dm, "The Depth of realspace and H_matrix is not matched!"
    realspace_padded = realspace
    try:
        realspace_padded = torch.nn.functional.pad(realspace_padded, pad=padding, value=0)
        padding = 'valid'
    except: pass
    # out = torch.nn.functional.conv2d(realspace_padded, H_mtx[None,...], None, padding='valid')
    realspace_perm = realspace_padded.permute([-1,0,1,2])   # (C,D,H,W)
    H_perm = H_mtx[None,...]                                # (1,Dm,Hm,Wm)
    # out = torch.nn.functional.conv2d(realspace_perm, H_perm, None, padding=padding).permute([1,2,3,0])[0]   # (H,W,C)
    out = fft_conv(realspace_perm, H_perm, None, padding=padding).permute([1, 2, 3, 0])[0]  # (H,W,C)
    return out

def back_project(projections:torch.Tensor, Hts:torch.Tensor):
    """
    Perform back projection, by deconvolution.
    """
    N,H,W,C = projections.shape
    Nm,Dm,Hm,Wm = Hts.shape
    assert N == Nm, "#Hts and # projections is not matched!"
    projections_perm = projections.permute([3,0,1,2])    # (C,N,H,W)
    Hts_perm = Hts.permute([1,0,2,3])   # (D,N,H,W)

    realspace = torch.nn.functional.conv2d(projections_perm, Hts_perm, None, padding='same')    # (C,D,H,W)


    realspace = realspace.permute([1,2,3,0]) / N
    return realspace



if __name__=='__main__':
    import tifffile
    from tqdm import trange
    from glbSettings import *
    from core.load import getH
    import imageio
    import os,re
    import torchvision.transforms.functional as TF
    print("Run rendering on CUDA: ", torch.cuda.is_available())
    torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else torch.set_default_tensor_type("torch.FloatTensor")

    # O_dir = r"E:\NetCode\Nerf_test\Simu\PSF\shujia_view\stack\deblur_35.tif"

    img_path= r'J:\Nerf_test\Data\20231214_tri_views_ER\paired\processed\EqualIntensity_ref_1'
    H_dir = r"J:\Nerf_test\Data\FLFM_View7_livingcell\FLFM_live\psf_step0.2\full2/full_hex7_60x_1.5.mat"
    out_dir = "./Matrix_psf_step0.2_noRect"
    N_step = 101


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # centercrops = [[1017,1200],[1017,1700],[1450,951],[1450,1450],[1450,1949],[1883,1200],[1883,1700]]
    centercrops=[[1017,1200],[1017,1700],[1450,951],[1450,1450],[1450,1949],[1883,1200],[1883,1700]]
    radius = 81//2

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    Hs = getH(H_dir, center = centercrops, radius = radius)
    np.save(out_dir + '/Hs_matrix', Hs)


    # O = torch.from_numpy(O).to(DEVICE)      # (D,H,W)
    Hs = torch.from_numpy(Hs).to(DEVICE)    # (D,H,W)

    # forward_project test
    costumed_psf=np.array([
        [0,0,0],
        [0, 1, 0],
        [0, 0, 0],
                           ],np.float32)

    costumed_psf= torch.from_numpy(costumed_psf).to(DEVICE)
    new_psf= torch.zeros_like(Hs)




    for _v in trange(Hs.shape[0]):
        local_psf=Hs[_v]
        loca_new_psf=new_psf[_v]
        max_list=[]
        for _d in trange(local_psf.shape[0]):
            max_idx=torch.argmax(local_psf[_d],keepdim=True)
            loca_new_psf[_d][max_idx//local_psf.shape[1],max_idx%local_psf.shape[2]]=1

        loca_new_psf = torch.nn.functional.conv2d(loca_new_psf.unsqueeze(1),costumed_psf[None,None,...],padding='same')
        new_psf[_v]=loca_new_psf.squeeze(1)




    ht = TF.rotate(Hs, angle=180)
    file_list=[]
    data_list=[]
    for _f in os.listdir(img_path):
        if re.search('.*.tif', _f):
            file_list.append(_f)
            file_path = os.path.join(img_path, _f)
            temp_img=imageio.imread(file_path)
            temp_img = np.asarray(temp_img, np.float32)
            data_list.append(temp_img)


    for _f,_img in zip(file_list,data_list):

        view_stack = torch.from_numpy(_img).to(DEVICE)
        if view_stack.dim()==3:
            view_stack=view_stack[...,None]
        # ht trans
        Ohat = back_project(view_stack, ht).to("cpu").numpy()

        # deconvolution
        intial_vol=torch.ones(Ohat.shape,dtype=torch.float32)
        for ii in range(N_step):
            erro_list=[]
            for view_idx in range(ht.shape[0]):
                fpj =forward_project(intial_vol, new_psf[view_idx])
                erro_back=view_stack[view_idx]/fpj
                erro_back[erro_back<0]=0
                erro_back[torch.isinf(erro_back)]=0
                erro_back[torch.isnan(erro_back)] = 0
                erro_list.append(erro_back)

            erro_list = torch.stack(erro_list, dim=0).to(DEVICE)
            bpjerro = back_project(erro_list, ht)
            intial_vol=bpjerro*intial_vol
            if ii%10==0:
                tifffile.imwrite(out_dir+'/Deconv_%s_%04d.tif'%(_f,ii), intial_vol[...,0].to("cpu").numpy())