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
    import os

    print("Run rendering on CUDA: ", torch.cuda.is_available())
    torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else torch.set_default_tensor_type("torch.FloatTensor")

    O_dir = r"J:\Nerf_test\Data\FLFM_View7_livingcell\FLFM_live\psf_step0.2\full_rect_center_0735\EqualIntensity_ref_1\LFP\Deconv_ori\Recon_viewstack_ite050.tif_ite050.tif"
    H_dir = r"J:\Nerf_test\Data\FLFM_View7_livingcell\FLFM_live\psf_step0.2\full_rect_center_0735/psf_matrix_rect.mat"
    out_dir = "./test_projection"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # standar hex7
    centercrops=[[1017,1200],[1017,1700],[1450,951],[1450,1450],[1450,1949],[1883,1200],[1883,1700]]

    # shujia hex3
    # centercrops = [[605, 500], [605, 1000], [1039, 750]]


    # centercrops=[[1017, 1200], [1450, 1450], [1450, 1949], [1883, 1200]]
    # centercrops = [[1017, 1200], [1450, 1949], [1883, 1200]]
    # centercrops = [[1450, 1450]]
    # centercrops = [[1450, 951], [1450, 1949]]
    # centercrops = [[1450, 951],[1450, 1450],  [1450, 1949]]
    # coordinates:
    # coordi_path = r'E:\NetCode\Nerf_test\LF_INR_MixedFusion\core\utils/coordi_dense19.mat'
    # centercrops = mat73.loadmat(coordi_path)['coordi']

    radius = 81//2


    # centercrops = [[603, 498], [603, 998], [1037, 748]]
    # radius = 81//2

    O = np.array(tifffile.imread(O_dir)[::-1,...,None], dtype=np.float32)
    Hs = getH(H_dir, center = centercrops, radius = radius)

    np.save(out_dir + '/Hs_matrix', Hs)


    O = torch.from_numpy(O).to(DEVICE)      # (D,H,W)
    Hs = torch.from_numpy(Hs).to(DEVICE)    # (D,H,W)

    # forward_project test

    costumed_psf=np.array([
        [0,0,0],
        [0, 1, 0],
        [0, 0, 0],
                           ],np.float32)
    # costumed_psf=imageio.imread(r'E:\NetCode\Nerf_test\LF_INR3\data\LF\view7_roi\psf\pattern.tif')
    costumed_psf=np.float32(costumed_psf)
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


    for _v in trange(new_psf.shape[0]):

        new_psf[_v]= new_psf[_v] / torch.sum(new_psf[_v])
        Hs[_v] = Hs[_v] / torch.sum(Hs[_v])


    projection_list = []
    new_projection_list=[]
    for i in trange(Hs.shape[0]):
        out = forward_project(O, Hs[i])
        new_out = forward_project(O, new_psf[i])
        # tifffile.imwrite(out_dir+f'{i:02}.tif', out.to("cpu").numpy())
        # imageio.imwrite()
        projection_list.append(out)
        new_projection_list.append(new_out)

    # back_project test
    outs = torch.stack(projection_list, dim=0).to(DEVICE)
    new_viewstack= torch.stack(new_projection_list, dim=0).to(DEVICE)
    #
    imageio.volwrite(os.path.join(out_dir,'renorm_psf_projection.tif'),np.squeeze(outs.to("cpu").numpy()))
    imageio.volwrite(os.path.join(out_dir,'customed_psf_projection.tif'),np.squeeze(new_viewstack.to("cpu").numpy()))

    # imageio.imwrite(os.path.join(out_dir,'renorm_psf_projection.tif'),np.squeeze(outs.to("cpu").numpy()))
    # imageio.imwrite(os.path.join(out_dir,'customed_psf_projection.tif'),np.squeeze(new_viewstack.to("cpu").numpy()))
    #
    np.save(os.path.join(out_dir,'new_psf.npy'),new_psf.cpu().numpy())
    Ohat = back_project(outs, Hs).to("cpu").numpy()
    tifffile.imwrite(out_dir+'/bp.tif', Ohat[...,0])





    #
    #
    # import torchvision.transforms.functional as TF
    # # ht trans
    # ht= TF.rotate(new_psf,angle=180)
    # # ht = ht.flip(1)
    #
    #
    #
    # Ohat = back_project(new_viewstack, ht).to("cpu").numpy()
    #
    # tifffile.imwrite(out_dir+'/bp.tif', Ohat[...,0])
    #
    #
    # # deconvolution
    # N_step=50
    # intial_vol=torch.ones(Ohat.shape,dtype=torch.float32)
    # for ii in range(N_step):
    #     erro_list=[]
    #     for view_idx in range(ht.shape[0]):
    #         fpj =forward_project(intial_vol, new_psf[view_idx])
    #         erro_back=new_viewstack[view_idx]/fpj
    #
    #         erro_back[erro_back<0]=0
    #         erro_back[torch.isinf(erro_back)]=0
    #         erro_back[torch.isnan(erro_back)] = 0
    #         erro_list.append(erro_back)
    #
    #     erro_list = torch.stack(erro_list, dim=0).to(DEVICE)
    #     bpjerro = back_project(erro_list, ht)
    #
    #     intial_vol=bpjerro*intial_vol
    #     if ii%10==0:
    #         tifffile.imwrite(out_dir+'/deconv_%04d.tif'%ii, intial_vol[...,0].to("cpu").numpy())