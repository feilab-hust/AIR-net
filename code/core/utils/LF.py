import torch
import numpy as np
from core.utils.fft_conv import fft_conv
import copy

def gen_scattered_vol(block:torch.Tensor, green_value:torch.Tensor, padding):
    new_block = block.clone()
    for local_depth in range(1, block.shape[0]):

            # 递归
            # block_slice = torch.nn.functional.pad(new_block[local_depth - 1][None, ...], pad=padding, value=0)
            block_slice = torch.nn.functional.pad(block[local_depth - 1][None, ...], pad=padding, value=0)


            green_slice = torch.nn.functional.pad(green_value[local_depth - 1][None, ..., None], pad=padding, value=0)
            temp_value = fft_conv(block_slice.squeeze(-1)[None, ...], green_slice.squeeze(-1)[None, ...],
                                  padding='same')
            temp_value = temp_value[0].unsqueeze(-1)

            dim_2_s = padding[2]
            dim_2_e = -padding[3] if padding[3] != 0 else temp_value.shape[2]

            dim_1_s = padding[4]
            dim_1_e = -padding[5] if padding[5] != 0 else temp_value.shape[1]
            # print('depth:%d --block_slice: %.3f --green_slice:%.3f --conv_slice:%.3f'%(local_depth,
            #                                                                            torch.mean(
            #                                                                                block_slice).cpu().data.numpy(),
            #                                                                            torch.mean(
            #                                                                                green_slice).cpu().data.numpy(),
            #                                                                            torch.mean(temp_value).cpu().data.numpy()
            #                                                                            )
            #       )
            #
            new_block[local_depth] = block[local_depth] + temp_value[:, dim_1_s:dim_1_e, dim_2_s:dim_2_e, :]
    return new_block
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
    import os

    print("Run rendering on CUDA: ", torch.cuda.is_available())
    torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else torch.set_default_tensor_type("torch.FloatTensor")

    O_dir = r"J:\Nerf_test\Data\20230426_clb_tube\registration\Seq_1\whole_fov\scale_padding\paired\stack\Padding_1_0001.tif"
    H_dir = r"E:\NetCode\Nerf_test\LF_INR3\data\LF\view7_roi/psf/View7_full_mat.mat"
    out_dir = "./Seq_gamma_tube1"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # centercrops = [[1017,1200],[1017,1700],[1450,951],[1450,1450],[1450,1949],[1883,1200],[1883,1700]]
    centercrops=[[1017,1200],[1017,1700],[1450,951],[1450,1450],[1450,1949],[1883,1200],[1883,1700]]
    radius = 81//2
    # O = np.array(tifffile.imread(O_dir)[::-1,...,None], dtype=np.float32)

    import imageio
    O = np.asarray(imageio.volread(O_dir)[::-1,...,None],np.float32)

    Hs = getH(H_dir, center = centercrops, radius = radius)

    Hs = np.asarray([_h / np.sum(_h) for _h in Hs])

    O = torch.from_numpy(O).to(DEVICE)      # (D,H,W)
    Hs = torch.from_numpy(Hs).to(DEVICE)    # (D,H,W)
    # forward_project test
    outs = []
    for i in trange(Hs.shape[0]):
        out = forward_project(O, Hs[i],padding='same')
        # tifffile.imwrite(out_dir+f'{i:02}.tif', out.to("cpu").numpy())
        outs.append(out)
    view_stack=torch.stack(outs,0)
    imageio.volwrite('viewstack.tif',np.squeeze(view_stack.data.cpu().numpy()))

    # back_project test
    outs = torch.stack(outs, dim=0).to(DEVICE)
    Ohat = back_project(outs, Hs).to("cpu").numpy()
    tifffile.imwrite(out_dir+'/_bp.tif', Ohat[...,0])

    #
    import torchvision.transforms.functional as TF

    new_viewstack=view_stack
    new_psf = Hs
    # ht trans
    ht = TF.rotate(new_psf, angle=180)
    # ht = ht.flip(1)
    Ohat = back_project(new_viewstack, ht).to("cpu").numpy()

    tifffile.imwrite(out_dir + '/bp.tif', Ohat[..., 0])

    # deconvolution
    N_step = 50
    intial_vol = torch.ones(Ohat.shape, dtype=torch.float32)
    for ii in range(N_step):
        erro_list = []
        for view_idx in range(ht.shape[0]):
            fpj = forward_project(intial_vol, new_psf[view_idx])
            erro_back = new_viewstack[view_idx] / fpj

            erro_back[erro_back < 0] = 0
            erro_back[torch.isinf(erro_back)] = 0
            erro_back[torch.isnan(erro_back)] = 0
            erro_list.append(erro_back)

        erro_list = torch.stack(erro_list, dim=0).to(DEVICE)
        bpjerro = back_project(erro_list, ht)

        intial_vol = bpjerro * intial_vol
        if ii % 10 == 0:
            tifffile.imwrite(out_dir + '/deconv_%04d.tif' % ii, intial_vol[..., 0].to("cpu").numpy())